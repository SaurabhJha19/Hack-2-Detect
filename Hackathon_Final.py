import cv2

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from collections import deque
import time
import warnings
from threading import Thread, Lock
from queue import Queue
import sqlite3
from datetime import datetime
import logging
from typing import Optional, Dict, List, Tuple, Any
from torch.amp import autocast
import torch.nn.functional as F
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
from logging.handlers import RotatingFileHandler

# Enhanced logging configuration with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        RotatingFileHandler('tracker.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class Config:
    """Centralized configuration management"""
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    half_precision: bool = torch.cuda.is_available()
    similarity_threshold: float = 0.75
    max_track_age: float = 15.0
    batch_size: int = 8
    buffer_size: int = 16
    target_fps: int = 30
    feature_dim: int = 2048
    motion_dim: int = 64
    frame_width: int = 640
    frame_height: int = 360

    def __post_init__(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

config = Config()

class MemoryPool:
    """Enhanced memory management with thread safety"""
    def __init__(self, max_size: int = 32):
        self.pool = Queue(maxsize=max_size)
        self.max_size = max_size
        self._lock = threading.Lock()

    def get(self, shape: tuple, dtype: np.dtype = np.uint8) -> np.ndarray:
        with self._lock:
            try:
                arr = self.pool.get_nowait()
                if arr.shape == shape and arr.dtype == dtype:
                    arr.fill(0)
                    return arr
            except:
                pass
            return np.zeros(shape, dtype=dtype)

    def put(self, arr: np.ndarray) -> None:
        with self._lock:
            try:
                self.pool.put_nowait(arr)
            except:
                pass

class VideoStream:
    """Optimized video capture with enhanced error handling and property management"""
    def __init__(self, source: Any, buffer_size: int = config.buffer_size):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
            
        self._configure_capture()
        
        self.queue = Queue(maxsize=buffer_size)
        self.lock = Lock()
        self.running = True
        self.frame_pool = MemoryPool()
        self.last_frame_time = 0
        self.frame_count = 0
        self.error_count = 0
        
        # Get actual capture properties after configuration
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1.0 / (self.actual_fps if self.actual_fps > 0 else config.target_fps)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video stream initialized - FPS: {self.actual_fps:.2f}, "
                   f"Resolution: {self.width}x{self.height}")
        
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Wait for first frame with timeout
        timeout = time.time() + 5.0
        while self.frame_count == 0 and time.time() < timeout:
            time.sleep(0.1)
        
        if self.frame_count == 0:
            raise RuntimeError(f"Failed to receive frames from source: {source}")

    def _configure_capture(self) -> None:
        """Configure capture properties with fallback values"""
        # Get original properties
        original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Only try to set properties if they're different from desired values
        if original_width != config.frame_width or original_height != config.frame_height:
            # Try to set resolution
            if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width):
                logger.info(f"Using original width: {original_width}")
            if not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height):
                logger.info(f"Using original height: {original_height}")
        
        # Try to set FPS if original is too different from target
        if abs(original_fps - config.target_fps) > 1:
            if not self.cap.set(cv2.CAP_PROP_FPS, config.target_fps):
                logger.info(f"Using original FPS: {original_fps}")
        
        # Try to minimize capture buffer
        if not self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1):
            logger.info("Using default buffer size")

    def _update(self) -> None:
        """Frame capture loop with improved timing and error handling"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_frame_time >= self.frame_interval:
                    if not self.queue.full():
                        ret, frame = self.cap.read()
                        if ret:
                            self.error_count = 0
                            
                            # Resize only if dimensions don't match
                            if frame.shape[1] != config.frame_width or frame.shape[0] != config.frame_height:
                                resized = self.frame_pool.get((config.frame_height, config.frame_width, 3))
                                cv2.resize(frame, (config.frame_width, config.frame_height), dst=resized)
                            else:
                                resized = frame.copy()
                            
                            with self.lock:
                                self.queue.put((current_time, resized))
                                self.frame_count += 1
                            self.last_frame_time = current_time
                        else:
                            self.error_count += 1
                            if self.error_count > 10:
                                logger.error(f"Multiple frame capture failures for source: {self.source}")
                                self._attempt_recovery()
                    time.sleep(0.001)  # Small sleep to prevent CPU overload
            except Exception as e:
                logger.error(f"Frame capture error: {str(e)}")
                self._attempt_recovery()

    def _attempt_recovery(self) -> None:
        """Attempt to recover from capture errors"""
        try:
            if isinstance(self.source, str):
                logger.info(f"Attempting to recover video capture for source: {self.source}")
                self.cap.release()
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    raise RuntimeError("Failed to reopen capture device")
                self._configure_capture()
                self.error_count = 0
                logger.info(f"Successfully recovered video capture for source: {self.source}")
            else:
                self.running = False
                logger.error(f"Unrecoverable capture error for source: {self.source}")
        except Exception as e:
            logger.error(f"Recovery attempt failed: {str(e)}")
            self.running = False

    def read(self) -> Optional[Tuple[float, np.ndarray]]:
        with self.lock:
            try:
                return self.queue.get_nowait() if not self.queue.empty() else None
            except:
                return None

    def stop(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
        
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break

class FeatureExtractor:
    """Optimized feature extraction with model caching and batching"""
    def __init__(self, batch_size: int = config.batch_size):
        self.batch_size = batch_size
        self.memory_pool = MemoryPool()
        self._initialize_models()
        self.transform = self._create_transform_pipeline()
        
        self.feature_dim = config.feature_dim
        self.motion_dim = config.motion_dim
        self.total_dim = self.feature_dim + self.motion_dim
        
        self.prev_frame = None
        self.feature_cache = {}
        self.cache_lock = Lock()
        
    def _initialize_models(self) -> None:
        self.person_detector = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        ).eval().to(config.device)
        
        self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor.fc = torch.nn.Identity()
        self.feature_extractor = self.feature_extractor.eval().to(config.device)
        
        if config.half_precision:
            self.person_detector = self.person_detector.half()
            self.feature_extractor = self.feature_extractor.half()

    def _create_transform_pipeline(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def detect_persons(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            img = torch.from_numpy(frame).permute(2, 0, 1)
            img = img.float().div(255.0).unsqueeze(0)
            
            if config.half_precision:
                img = img.half()
            
            img = img.to(config.device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.half_precision):
                predictions = self.person_detector(img)[0]
            
            mask = (predictions['labels'] == 1) & (predictions['scores'] > 0.7)
            if not mask.any():
                return None
                
            return predictions['boxes'][mask].cpu().numpy()
            
        except Exception as e:
            logger.error(f"Person detection error: {str(e)}")
            return None

    @torch.no_grad()
    def get_features(self, frame: np.ndarray, boxes: np.ndarray) -> Optional[torch.Tensor]:
        try:
            if boxes is None or len(boxes) == 0:
                return None
                
            features_list = []
            
            for i in range(0, len(boxes), self.batch_size):
                batch_boxes = boxes[i:i + self.batch_size]
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.half_precision):
                    appearance = self.get_appearance_features(frame, batch_boxes)
                    if appearance is None:
                        continue
                        
                    motion = self.get_motion_features(frame, batch_boxes)
                    motion_tensor = torch.from_numpy(motion).float().to(config.device)
                    
                    if config.half_precision:
                        motion_tensor = motion_tensor.half()
                        
                    # Ensure consistent feature dimensions
                    appearance = appearance.view(len(batch_boxes), -1)  # Flatten if needed
                    motion_tensor = motion_tensor.view(len(batch_boxes), -1)  # Flatten if needed
                    
                    batch_features = torch.cat([appearance, motion_tensor], dim=1)
                    features_list.append(batch_features)
            
            if not features_list:
                return None
                
            # Ensure all features have the same dimensions
            concatenated_features = torch.cat(features_list, dim=0)
            expected_dim = self.feature_dim + self.motion_dim
            if concatenated_features.shape[1] != expected_dim:
                logger.warning(f"Feature dimension mismatch. Expected {expected_dim}, got {concatenated_features.shape[1]}")
                return None
                
            return concatenated_features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return None

    def get_appearance_features(self, frame: np.ndarray, boxes: np.ndarray) -> Optional[torch.Tensor]:
        try:
            patches = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                patch = frame[y1:y2, x1:x2]
                if patch.size > 0:
                    tensor = self.transform(patch)
                    if config.half_precision:
                        tensor = tensor.half()
                    patches.append(tensor)
            
            if not patches:
                return None
                
            batch = torch.stack(patches).to(config.device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.half_precision):
                return self.feature_extractor(batch)
                
        except Exception as e:
            logger.error(f"Appearance feature extraction error: {str(e)}")
            return None

    def get_motion_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        try:
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_frame is None:
                self.prev_frame = current
                return np.zeros((len(boxes), self.motion_dim))
            
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, current,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            motion_features = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                roi_flow = flow[y1:y2, x1:x2]
                
                if roi_flow.size > 0:
                    hist = np.histogram2d(
                        roi_flow[..., 0].flatten(),
                        roi_flow[..., 1].flatten(),
                        bins=8
                    )[0].flatten()
                    
                    if len(hist) < self.motion_dim:
                        hist = np.pad(hist, (0, self.motion_dim - len(hist)))
                    else:
                        hist = hist[:self.motion_dim]
                        
                    motion_features.append(hist)
                else:
                    motion_features.append(np.zeros(self.motion_dim))
            
            self.prev_frame = current
            return np.array(motion_features)
            
        except Exception as e:
            logger.error(f"Motion feature extraction error: {str(e)}")
            return np.zeros((len(boxes), self.motion_dim))

class TrackedPerson:
    """Enhanced person tracking with improved state management"""
    def __init__(self, person_id: int, initial_features: torch.Tensor,
                 box: np.ndarray, camera_id: int):
        self.id = person_id
        self.feature_queue = deque(maxlen=30)
        self.feature_queue.append(initial_features)
        self.last_seen = time.time()
        self.last_camera_id = camera_id
        self.color = np.random.randint(0, 255, 3).tolist()
        self.location_history = deque(maxlen=50)
        self.camera_history = deque(maxlen=50)
        self.velocity = np.zeros(2)
        self.update_location(box, camera_id)
        
    def update_features(self, features: torch.Tensor) -> None:
        self.feature_queue.append(features)
        self.last_seen = time.time()
        
    def update_location(self, box: np.ndarray, camera_id: int) -> None:
        if len(self.location_history) > 0:
            prev_box = self.location_history[-1]
            prev_center = ((prev_box[0] + prev_box[2]) / 2, 
                          (prev_box[1] + prev_box[3]) / 2)
            curr_center = ((box[0] + box[2]) / 2, 
                          (box[1] + box[3]) / 2)
            
            new_velocity = np.array([
                curr_center[0] - prev_center[0],
                curr_center[1] - prev_center[1]
            ])
            self.velocity = 0.7 * self.velocity + 0.3 * new_velocity
            
        self.location_history.append(box.astype(np.float32))
        self.camera_history.append(camera_id)
        self.last_camera_id = camera_id
        
    def get_avg_features(self) -> torch.Tensor:
        return torch.mean(torch.stack(list(self.feature_queue)), dim=0)

class DatabaseManager:
    """Enhanced database management with thread-local connections"""
    def __init__(self, db_path: str = 'tracking_database.db'):
        self.db_path = db_path
        self._local = threading.local()
        self.pool_lock = Lock()
        
        # Initialize database schema
        with self.get_connection() as conn:
            self.setup_database(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection for the current thread"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_connection(self):
        """Get or create a thread-local connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = self._create_connection()
        return ConnectionContext(self._local.connection)

    def setup_database(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema with indexes"""
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS tracked_persons (
                person_id INTEGER PRIMARY KEY,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feature_vector BLOB NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS location_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                camera_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                x1 FLOAT NOT NULL,
                y1 FLOAT NOT NULL,
                x2 FLOAT NOT NULL,
                y2 FLOAT NOT NULL,
                confidence FLOAT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES tracked_persons(person_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_person_id ON location_history(person_id);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON location_history(timestamp);
            CREATE INDEX IF NOT EXISTS idx_camera_location ON location_history(camera_id, timestamp);
        ''')
        conn.commit()

    def add_person(self, features: np.ndarray) -> int:
        """Add new person with thread-safe connection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO tracked_persons (feature_vector)
                    VALUES (?)
                ''', (features.tobytes(),))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Database insert error: {str(e)}")
            return -1

    def update_location(self, person_id: int, camera_id: int, 
                       box: np.ndarray, confidence: float) -> None:
        """Update person location with thread-safe connection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO location_history 
                    (person_id, camera_id, x1, y1, x2, y2, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (person_id, camera_id, 
                      float(box[0]), float(box[1]), 
                      float(box[2]), float(box[3]),
                      float(confidence)))
                
                cursor.execute('''
                    UPDATE tracked_persons 
                    SET last_seen = CURRENT_TIMESTAMP 
                    WHERE person_id = ?
                ''', (person_id,))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Location update error: {str(e)}")

    def get_person_history(self, person_id: int, time_window: int = 3600) -> List[Dict]:
        """Retrieve person history with thread-safe connection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT camera_id, timestamp, x1, y1, x2, y2, confidence
                    FROM location_history
                    WHERE person_id = ?
                    AND timestamp >= datetime('now', '-' || ? || ' seconds')
                    ORDER BY timestamp DESC
                ''', (person_id, time_window))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"History retrieval error: {str(e)}")
            return []

class ConnectionContext:
    """Context manager for thread-local database connections"""
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.connection.rollback()
        else:
            try:
                self.connection.commit()
            except sqlite3.Error:
                self.connection.rollback()

class MultiCameraTracker:
    def __init__(self, similarity_thresh: float = config.similarity_threshold, 
                 batch_size: int = config.batch_size):
        self.feature_extractor = FeatureExtractor(batch_size=batch_size)
        self.tracks: Dict[int, TrackedPerson] = {}
        self.db_manager = DatabaseManager()
        self.lock = threading.RLock()  # Changed to RLock for reentrant locking
        self.similarity_thresh = similarity_thresh
        self.camera_buffers: Dict[int, deque] = {}
        self.max_cosine_sim = torch.nn.CosineSimilarity(dim=1)
        self.frame_pool = MemoryPool()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.last_cleanup = time.time()
        
        # Selected person tracking with thread safety
        self._selected_person_id = None
        self._selecting_mode = False
        self.last_detected_boxes = {}
        self.last_frame_features = {}
        self.selection_event = threading.Event()
        
        # Cache for feature computation
        self.feature_cache = {}
        self.cache_lock = threading.Lock()
        
        # Camera state management
        self.camera_states = {}
        self.camera_locks = {}
        self.track_history = {}  # Add track history for visualization
        self.max_history_length = 30  # Maximum number of positions to store

        # Add new camera-specific tracking state
        self.current_tracking_camera = None  # Track which camera currently has the person
        self.camera_transition_cooldown = 1.0  # Seconds to wait before allowing camera transition
        self.last_camera_transition = 0
        self.last_confident_detection = None
        self.transition_threshold = 0.85

    def register_camera(self, camera_id: int) -> None:
        """Register new camera with optimized buffer and state management"""
        with self.lock:
            # Initialize camera-specific data structures
            self.camera_buffers[camera_id] = deque(maxlen=300)
            self.last_detected_boxes[camera_id] = None
            self.last_frame_features[camera_id] = None
            self.camera_locks[camera_id] = threading.Lock()
            self.camera_states[camera_id] = {
                'last_update': time.time(),
                'frame_count': 0,
                'processing_time': deque(maxlen=100),
                'detection_count': 0
            }
            
            # Clear any existing cache for this camera
            with self.cache_lock:
                keys_to_remove = [k for k in self.feature_cache if k[1] == camera_id]
                for k in keys_to_remove:
                    self.feature_cache.pop(k, None)
                    
            logger.info(f"Registered camera {camera_id} with optimized buffers")

    def get_camera_stats(self, camera_id: int) -> Dict[str, float]:
        """Get performance statistics for a specific camera"""
        with self.lock:
            if camera_id not in self.camera_states:
                return {}
                
            state = self.camera_states[camera_id]
            processing_times = state['processing_time']
            
            if not processing_times:
                return {'fps': 0, 'latency': 0, 'detection_rate': 0}
                
            avg_time = sum(processing_times) / len(processing_times)
            return {
                'fps': 1.0 / avg_time if avg_time > 0 else 0,
                'latency': avg_time * 1000,  # Convert to milliseconds
                'detection_rate': state['detection_count'] / max(1, state['frame_count'])
            }

    def update_camera_stats(self, camera_id: int, processing_time: float, 
                          detections: bool = False) -> None:
        """Update performance statistics for a specific camera"""
        with self.lock:
            if camera_id in self.camera_states:
                state = self.camera_states[camera_id]
                state['last_update'] = time.time()
                state['frame_count'] += 1
                state['processing_time'].append(processing_time)
                if detections:
                    state['detection_count'] += 1
        
    @property
    def selecting_mode(self):
        return self._selecting_mode
        
    @selecting_mode.setter
    def selecting_mode(self, value):
        with self.lock:
            self._selecting_mode = value
            
    @property
    def selected_person_id(self):
        return self._selected_person_id
        
    @selected_person_id.setter
    def selected_person_id(self, value):
        with self.lock:
            self._selected_person_id = value

    def select_person_at_point(self, camera_id: int, point_x: int, point_y: int) -> Optional[int]:
        """Select person with improved initialization"""
        try:
            with self.lock:
                if camera_id not in self.last_detected_boxes or camera_id not in self.last_frame_features:
                    return None
                    
                boxes = self.last_detected_boxes[camera_id]
                features = self.last_frame_features[camera_id]
                
                if boxes is None or features is None or len(boxes) == 0:
                    return None
                
                # Find all boxes containing the clicked point
                x1, y1, x2, y2 = boxes.T
                mask = (x1 <= point_x) & (point_x <= x2) & (y1 <= point_y) & (point_y <= y2)
                matching_indices = np.where(mask)[0]
                
                if len(matching_indices) == 0:
                    return None
                    
                # Use the first matching box
                idx = matching_indices[0]
                feature_vector = features[idx].cpu().numpy()
                
                # Generate person ID and initialize tracking
                person_id = self.db_manager.add_person(feature_vector)
                if person_id > 0:
                    # Initialize track with current frame's information
                    self.tracks[person_id] = TrackedPerson(
                        person_id, features[idx], boxes[idx], camera_id)
                    self.selected_person_id = person_id
                    
                    # Initialize track history
                    self.track_history[person_id] = deque(maxlen=self.max_history_length)
                    center = (
                        int((boxes[idx][0] + boxes[idx][2]) / 2),
                        int((boxes[idx][1] + boxes[idx][3]) / 2)
                    )
                    self.track_history[person_id].append(center)
                    
                    self.selection_event.set()
                    logger.info(f"Selected person {person_id} for tracking")
                    return person_id
                    
                return None
                
        except Exception as e:
            logger.error(f"Person selection error: {str(e)}")
            return None
            
    def update_selected_person_tracking(self, features: torch.Tensor, boxes: np.ndarray, camera_id: int) -> None:
        """Update tracking with proper camera transition handling"""
        try:
            with self.lock:
                if self.selected_person_id is None or self.selected_person_id not in self.tracks:
                    return

                selected_track = self.tracks[self.selected_person_id]
                current_time = time.time()
                
                # Handle camera transition logic
                if self.current_tracking_camera is None:
                    self.current_tracking_camera = camera_id
                elif self.current_tracking_camera != camera_id:
                    # Only process other cameras if enough time has passed since last transition
                    if current_time - self.last_camera_transition < self.camera_transition_cooldown:
                        return
                
                # Get target features with temporal averaging
                target_features = selected_track.get_avg_features()
                
                # Compute appearance similarity scores
                similarities = self.max_cosine_sim(features, target_features.unsqueeze(0))
                
                # Initialize position-based weights
                position_weights = np.ones(len(boxes))
                
                # Add position-based matching if we have history and we're in the same camera
                if len(selected_track.location_history) > 0 and camera_id == selected_track.last_camera_id:
                    last_box = selected_track.location_history[-1]
                    last_center = np.array([
                        (last_box[0] + last_box[2]) / 2,
                        (last_box[1] + last_box[3]) / 2
                    ])
                    
                    current_centers = np.array([
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                        for box in boxes
                    ])
                    
                    distances = np.linalg.norm(current_centers - last_center, axis=1)
                    position_weights = 1 / (1 + 0.1 * distances)
                    position_weights = position_weights / np.max(position_weights)
                
                # Combine appearance and position scores
                combined_scores = 0.7 * similarities.cpu().numpy() + 0.3 * position_weights
                max_idx = np.argmax(combined_scores)
                max_sim = combined_scores[max_idx]

                # Different thresholds for same-camera and cross-camera tracking
                threshold = (
                    self.transition_threshold 
                    if camera_id != self.current_tracking_camera 
                    else self.similarity_thresh
                )
                
                # Update tracking if similarity is high enough
                if max_sim > threshold:
                    matched_box = boxes[max_idx]
                    
                    # Handle camera transition
                    if camera_id != self.current_tracking_camera:
                        # Only transition if we have a very confident match
                        if max_sim > self.transition_threshold:
                            logger.info(f"Camera transition: {self.current_tracking_camera} -> {camera_id}")
                            self.current_tracking_camera = camera_id
                            self.last_camera_transition = current_time
                            # Clear motion history for new camera
                            if self.selected_person_id in self.track_history:
                                self.track_history[self.selected_person_id].clear()
                    
                    # Update features and location
                    selected_track.update_features(features[max_idx])
                    selected_track.update_location(matched_box, camera_id)
                    
                    # Update track visualization history only for current camera
                    if camera_id == self.current_tracking_camera:
                        if self.selected_person_id not in self.track_history:
                            self.track_history[self.selected_person_id] = deque(maxlen=self.max_history_length)
                        
                        center = (
                            int((matched_box[0] + matched_box[2]) / 2),
                            int((matched_box[1] + matched_box[3]) / 2)
                        )
                        self.track_history[self.selected_person_id].append(center)
                    
                    # Update last confident detection
                    self.last_confident_detection = {
                        'camera_id': camera_id,
                        'time': current_time,
                        'similarity': max_sim
                    }
                    
                    # Async database update
                    threading.Thread(
                        target=self.db_manager.update_location,
                        args=(self.selected_person_id, camera_id, matched_box, float(max_sim)),
                        daemon=True
                    ).start()

        except Exception as e:
            logger.error(f"Selected person tracking update error: {str(e)}")

    def process_frame(self, frame: np.ndarray, camera_id: int) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
            
        start_time = time.time()
        try:
            # Convert to RGB and store frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.camera_buffers[camera_id].append({
                'frame': frame.copy(),
                'timestamp': time.time()
            })
            
            # Person detection
            boxes = self.feature_extractor.detect_persons(rgb_frame)
            
            if boxes is None or len(boxes) == 0:
                self.last_detected_boxes[camera_id] = None
                self.last_frame_features[camera_id] = None
                return frame
            
            # Feature extraction
            features = self.feature_extractor.get_features(rgb_frame, boxes)
            
            if features is None:
                self.last_detected_boxes[camera_id] = None
                self.last_frame_features[camera_id] = None
                return frame
            
            # Update detection results
            with self.lock:
                self.last_detected_boxes[camera_id] = boxes.copy()
                self.last_frame_features[camera_id] = features.clone()
            
            # If in selection mode or no person selected, show all detections
            if self.selecting_mode or self.selected_person_id is None:
                return self.draw_all_detections(frame, boxes, camera_id)
            
            # Update tracking immediately instead of in a separate thread
            self.update_selected_person_tracking(features, boxes, camera_id)
            
            # Get the most recent tracking state for visualization
            processed_frame = self.draw_tracking(frame, boxes, camera_id)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return frame
        finally:
            self.processing_times.append(time.time() - start_time)

    def draw_all_detections(self, frame: np.ndarray, boxes: np.ndarray, camera_id: int) -> np.ndarray:
        """Draw all detected persons with selection instructions"""
        try:
            vis_frame = self.frame_pool.get(frame.shape)
            np.copyto(vis_frame, frame)
            
            # Draw camera ID and mode information
            mode_text = "SELECTION MODE: Click on a person to track" if self.selecting_mode else "Press 'p' to enter selection mode"
            cv2.putText(vis_frame, f"Camera {camera_id} | {mode_text}", 
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Draw all detected boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            return vis_frame
            
        except Exception as e:
            logger.error(f"Detection visualization error: {str(e)}")
            return frame

    def draw_tracking(self, frame: np.ndarray, boxes: np.ndarray, camera_id: int) -> np.ndarray:
        """Enhanced visualization with camera-specific tracking status"""
        try:
            vis_frame = self.frame_pool.get(frame.shape)
            np.copyto(vis_frame, frame)

            if self.selected_person_id in self.tracks:
                track = self.tracks[self.selected_person_id]
                
                # Only draw tracking visualization for current tracking camera
                if camera_id == self.current_tracking_camera:
                    if len(track.location_history) > 0:
                        latest_box = track.location_history[-1]
                        x1, y1, x2, y2 = map(int, latest_box)
                        
                        time_since_update = time.time() - track.last_seen
                        
                        confidence_color = list(track.color)
                        if time_since_update > 0.5:
                            alpha = max(0, 1 - (time_since_update - 0.5))
                            confidence_color = [int(c * alpha) for c in track.color]
                        
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), confidence_color, 2)
                        
                        label = f"ID: {track.id} | {time_since_update:.1f}s"
                        cv2.putText(vis_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)
                        
                        if self.selected_person_id in self.track_history:
                            history = list(self.track_history[self.selected_person_id])
                            if len(history) > 1:
                                for i in range(1, len(history)):
                                    alpha = i / len(history)
                                    color = [int(c * alpha) for c in track.color]
                                    pt1 = history[i - 1]
                                    pt2 = history[i]
                                    cv2.line(vis_frame, pt1, pt2, color, 2)

            # Draw status overlay with camera-specific information
            if camera_id == self.current_tracking_camera:
                status = f"Actively tracking ID: {self.selected_person_id}"
            else:
                status = f"Monitoring for ID: {self.selected_person_id}"
            
            cv2.putText(vis_frame, f"Camera {camera_id} | {status}",
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            return vis_frame

        except Exception as e:
            logger.error(f"Tracking visualization error: {str(e)}")
            return frame

    def clean_old_tracks(self) -> None:
        """Remove old tracks that haven't been updated recently"""
        current_time = time.time()
        with self.lock:
            to_remove = []
            for track_id, track in self.tracks.items():
                if current_time - track.last_seen > config.max_track_age:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                if track_id != self.selected_person_id:  # Don't remove selected person
                    self.tracks.pop(track_id, None)

    def get_performance_stats(self) -> Dict[str, float]:
        """Get tracking performance statistics"""
        if not self.processing_times:
            return {'fps': 0, 'latency': 0}
            
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        return {
            'fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'latency': avg_processing_time * 1000  # Convert to milliseconds
        }
    def toggle_selecting_mode(self) -> None:
        """Toggle person selection mode with thread safety"""
        with self.lock:
            self._selecting_mode = not self._selecting_mode
            if self._selecting_mode:
                # Reset selection state when entering selection mode
                self.selection_event.clear()
                logger.info("Person selection mode activated. Click on a person to track.")
            else:
                # Clear any pending selections when exiting selection mode
                self.selection_event.set()
                logger.info("Person selection mode deactivated.")

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function for person selection"""
    tracker = param['tracker']
    camera_id = param['camera_id']
    
    if event == cv2.EVENT_LBUTTONDOWN and tracker.selecting_mode:
        selected_id = tracker.select_person_at_point(camera_id, x, y)
        if selected_id is not None:
            tracker.selecting_mode = False

def main():
    """Main function demonstrating multi-camera tracking"""
    try:
        # Initialize video sources
        sources = {
            #0: 0#,  # Default camera
            # Add more cameras as needed
            1: r"D:\Codes\gettyimages-1126783091-640_adpp.mp4",
            2: r"D:\Codes\gettyimages-1221865450-640_adpp.mp4"
            #3: r""
        }
        
        streams = {}
        tracker = MultiCameraTracker()
        
        # Initialize windows and callbacks
        for camera_id, source in sources.items():
            try:
                streams[camera_id] = VideoStream(source)
                tracker.register_camera(camera_id)
                
                window_name = f"Camera {camera_id}"
                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, mouse_callback, 
                                   {'tracker': tracker, 'camera_id': camera_id})
                
            except Exception as e:
                logger.error(f"Failed to initialize camera {camera_id}: {str(e)}")
        
        if not streams:
            raise RuntimeError("No cameras could be initialized")
        
        while True:
            for camera_id, stream in streams.items():
                frame_data = stream.read()
                if frame_data is not None:
                    timestamp, frame = frame_data
                    
                    # Process and display frame
                    processed_frame = tracker.process_frame(frame, camera_id)
                    if processed_frame is not None:
                        cv2.imshow(f"Camera {camera_id}", processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                tracker.toggle_selecting_mode()
            
            # Monitor system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            if cpu_percent > 80 or memory_percent > 80:
                logger.warning(f"High resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%")
        
    except Exception as e:
        logger.error(f"Main loop error: {str(e)}")
    
    finally:
        # Cleanup
        for stream in streams.values():
            stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()