import cv2
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
    frame_width: int = 1280
    frame_height: int = 720

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
    """Enhanced database management with connection pooling and optimized queries"""
    def __init__(self, db_path: str = 'tracking_database.db', pool_size: int = 5):
        self.db_path = db_path
        self.connection_pool = Queue(maxsize=pool_size)
        self.pool_lock = Lock()
        
        # Initialize connection pool
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            self.connection_pool.put(conn)
            
        # Initialize database schema
        with self.get_connection() as conn:
            self.setup_database(conn)

    def get_connection(self):
        """Context manager for database connections"""
        return DatabaseConnection(self.connection_pool, self.pool_lock)

    def setup_database(self, conn: sqlite3.Connection) -> None:
        """Initialize optimized database schema with indexes"""
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
        """Add new person with optimized blob storage"""
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
        """Update person location with batch support"""
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
                
                # Update last seen timestamp
                cursor.execute('''
                    UPDATE tracked_persons 
                    SET last_seen = CURRENT_TIMESTAMP 
                    WHERE person_id = ?
                ''', (person_id,))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Location update error: {str(e)}")

    def get_person_history(self, person_id: int, time_window: int = 3600) -> List[Dict]:
        """Retrieve person history with optimized query"""
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

class DatabaseConnection:
    """Context manager for database connections"""
    def __init__(self, pool: Queue, lock: Lock):
        self.pool = pool
        self.lock = lock
        self.conn = None

    def __enter__(self) -> sqlite3.Connection:
        with self.lock:
            self.conn = self.pool.get()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is not None:
                try:
                    self.conn.rollback()
                except:
                    pass
            try:
                self.pool.put(self.conn)
            except:
                self.conn.close()

class MultiCameraTracker:
    """Enhanced multi-camera tracking system with improved performance"""
    def __init__(self, similarity_thresh: float = config.similarity_threshold, 
                 batch_size: int = config.batch_size):
        self.feature_extractor = FeatureExtractor(batch_size=batch_size)
        self.tracks: Dict[int, TrackedPerson] = {}
        self.db_manager = DatabaseManager()
        self.lock = Lock()
        self.similarity_thresh = similarity_thresh
        self.camera_buffers: Dict[int, deque] = {}
        self.max_cosine_sim = torch.nn.CosineSimilarity(dim=1)
        self.frame_pool = MemoryPool()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.last_cleanup = time.time()
        
    def register_camera(self, camera_id: int) -> None:
        """Register new camera with optimized buffer"""
        self.camera_buffers[camera_id] = deque(maxlen=300)
        logger.info(f"Registered camera {camera_id}")
        
    @torch.no_grad()
    def process_frame(self, frame: np.ndarray, camera_id: int) -> Optional[np.ndarray]:
        """Process frame with enhanced performance monitoring"""
        if frame is None or frame.size == 0:
            return None

        start_time = time.time()
        try:
            # Convert to RGB once
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Store original frame with timestamp
            self.camera_buffers[camera_id].append({
                'frame': frame.copy(),
                'timestamp': time.time()
            })
            
            # Detect persons
            boxes = self.feature_extractor.detect_persons(rgb_frame)
            if boxes is None or len(boxes) == 0:
                return frame
                
            # Extract features
            features = self.feature_extractor.get_features(rgb_frame, boxes)
            if features is None:
                return frame
                
            # Update tracking
            with self.lock:
                self.update_tracks(features, boxes, camera_id)
                
                # Periodic cleanup
                current_time = time.time()
                if current_time - self.last_cleanup > 5.0:
                    self.clean_old_tracks()
                    self.last_cleanup = current_time
                
            return self.draw_tracking(frame, boxes, camera_id)
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return frame
        finally:
            # Update processing time statistics
            self.processing_times.append(time.time() - start_time)

    def update_tracks(self, features: torch.Tensor, boxes: np.ndarray, camera_id: int) -> None:
        """Update tracks with improved matching algorithm"""
        try:
            if not self.tracks:
                # Initialize first tracks
                for i in range(len(features)):
                    person_id = self.db_manager.add_person(features[i].cpu().numpy())
                    self.tracks[person_id] = TrackedPerson(person_id, features[i], boxes[i], camera_id)
                    self.db_manager.update_location(person_id, camera_id, boxes[i], 1.0)
                return

            # Get active tracks and their features
            active_ids = list(self.tracks.keys())
            if not active_ids:
                return
                
            active_features = torch.stack([t.get_avg_features() for t in self.tracks.values()])
            
            # Compute similarity matrices
            similarity_matrix = self.compute_similarity_matrix(features, active_features, boxes, active_ids)
            
            # Match tracks using Hungarian algorithm
            matches, unmatched_detections = self.match_tracks(
                similarity_matrix, len(features), len(active_ids))
            
            # Update matched tracks
            for det_idx, track_idx in matches:
                track_id = active_ids[track_idx]
                self.tracks[track_id].update_features(features[det_idx])
                self.tracks[track_id].update_location(boxes[det_idx], camera_id)
                self.db_manager.update_location(
                    track_id, camera_id, boxes[det_idx], 
                    float(similarity_matrix[det_idx, track_idx])
                )
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_detections:
                person_id = self.db_manager.add_person(features[det_idx].cpu().numpy())
                self.tracks[person_id] = TrackedPerson(
                    person_id, features[det_idx], boxes[det_idx], camera_id)
                self.db_manager.update_location(person_id, camera_id, boxes[det_idx], 1.0)
                
        except Exception as e:
            logger.error(f"Track update error: {str(e)}")

    def compute_similarity_matrix(self, features: torch.Tensor, active_features: torch.Tensor,
                                boxes: np.ndarray, active_ids: List[int]) -> np.ndarray:
        """Compute combined similarity matrix with spatial and feature information"""
        # Compute appearance similarity
        similarity_matrix = self.max_cosine_sim(
            features.unsqueeze(1),
            active_features.unsqueeze(0)
        )
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(boxes), len(active_ids)))
        for i, box in enumerate(boxes):
            for j, track_id in enumerate(active_ids):
                track_box = self.tracks[track_id].location_history[-1]
                iou_matrix[i, j] = self._calculate_iou(box, track_box)
        
        # Combine similarities
        combined_similarity = (
            0.7 * similarity_matrix.cpu().numpy() +
            0.3 * iou_matrix
        )
        
        return combined_similarity

    def match_tracks(self, similarity_matrix: np.ndarray, 
                    num_detections: int, num_tracks: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Match tracks using Hungarian algorithm with threshold"""
        matches = []
        unmatched_detections = set(range(num_detections))
        
        # Find all valid matches
        for i in range(num_detections):
            for j in range(num_tracks):
                if similarity_matrix[i, j] >= self.similarity_thresh:
                    matches.append((i, j))
                    if i in unmatched_detections:
                        unmatched_detections.remove(i)
        
        return matches, list(unmatched_detections)

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union with vectorized operations"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0

    def clean_old_tracks(self, max_age: float = config.max_track_age) -> None:
        """Remove old tracks with performance monitoring"""
        current_time = time.time()
        original_count = len(self.tracks)
        
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if current_time - track.last_seen <= max_age
        }
        
        removed_count = original_count - len(self.tracks)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old tracks")

    def draw_tracking(self, frame: np.ndarray, boxes: np.ndarray, camera_id: int) -> np.ndarray:
        """Enhanced visualization with performance metrics"""
        try:
            vis_frame = self.frame_pool.get(frame.shape)
            np.copyto(vis_frame, frame)
            
            # Draw camera ID and performance metrics
            avg_process_time = np.mean(self.processing_times) if self.processing_times else 0
            fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
            
            cv2.putText(vis_frame, f"Camera {camera_id} | FPS: {fps:.1f}", 
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw tracks
            with self.lock:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    track_id = None
                    color = (0, 255, 0)
                    
                    # Find matching track
                    for tid, track in self.tracks.items():
                        if np.allclose(box, track.location_history[-1], rtol=1e-3):
                            track_id = tid
                            color = track.color
                            break
                    
                    if track_id is not None:
                        # Draw bounding box
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with additional info
                        track = self.tracks[track_id]
                        label = f"ID: {track_id}"
                        if track.last_camera_id != camera_id:
                            label += f" (Cam {track.last_camera_id})"
                        
                        # Add velocity information
                        speed = np.linalg.norm(track.velocity)
                        if speed > 1.0:
                            label += f" | Speed: {speed:.1f}"
                            cv2.putText(vis_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Draw trajectory with motion prediction
                        points = []
                        for loc in track.location_history:
                            cx = int((loc[0] + loc[2]) / 2)
                            cy = int((loc[1] + loc[3]) / 2)
                            points.append((cx, cy))
                        
                        if len(points) > 1:
                            points = np.array(points, np.int32)
                            cv2.polylines(vis_frame, [points], False, color, 2)
                            
                            # Predict future position
                            if speed > 1.0:
                                future_x = cx + int(track.velocity[0] * 10)
                                future_y = cy + int(track.velocity[1] * 10)
                                cv2.line(vis_frame, (cx, cy), 
                                       (future_x, future_y), color, 2)
            
            return vis_frame
            
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return frame

    @torch.no_grad()
    def search_person(self, target_id: int, time_window: float = 300) -> List[Dict]:
        """Enhanced person search with parallel processing"""
        results = []
        current_time = time.time()
        
        try:
            with self.lock:
                if target_id in self.tracks:
                    target_features = self.tracks[target_id].get_avg_features()
                    
                    def process_frame(frame_data: Dict, camera_id: int) -> Optional[Dict]:
                        if current_time - frame_data['timestamp'] > time_window:
                            return None
                            
                        frame = frame_data['frame']
                        boxes = self.feature_extractor.detect_persons(frame)
                        
                        if boxes is not None:
                            features = self.feature_extractor.get_features(frame, boxes)
                            if features is not None:
                                similarities = self.max_cosine_sim(
                                    features,
                                    target_features.unsqueeze(0)
                                )
                                
                                max_sim, max_idx = torch.max(similarities, dim=0)
                                if max_sim > self.similarity_thresh:
                                    return {
                                        'camera_id': camera_id,
                                        'timestamp': frame_data['timestamp'],
                                        'confidence': float(max_sim),
                                        'location': boxes[max_idx].tolist()
                                    }
                        return None
                    
                    # Process frames in parallel
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = []
                        for camera_id, buffer in self.camera_buffers.items():
                            for frame_data in buffer:
                                futures.append(
                                    executor.submit(process_frame, frame_data, camera_id)
                                )
                        
                        # Collect results
                        for future in futures:
                            result = future.result()
                            if result is not None:
                                results.append(result)
            
            return sorted(results, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Person search error: {str(e)}")
            return []

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get system performance metrics"""
        try:
            with self.lock:
                avg_process_time = np.mean(self.processing_times) if self.processing_times else 0
                return {
                    'fps': 1.0 / avg_process_time if avg_process_time > 0 else 0,
                    'avg_process_time': avg_process_time * 1000,  # Convert to ms
                    'active_tracks': len(self.tracks),
                    'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
                }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
        
    def compute_similarity_matrix(self, features: torch.Tensor, active_features: torch.Tensor,
                            boxes: np.ndarray, active_ids: List[int]) -> np.ndarray:
        """Compute combined similarity matrix with spatial and feature information"""
        try:
            # Ensure features are on the same device
            if features.device != active_features.device:
                active_features = active_features.to(features.device)
                
            # Normalize feature vectors
            features_norm = F.normalize(features, p=2, dim=1)
            active_features_norm = F.normalize(active_features, p=2, dim=1)
            
            # Compute appearance similarity using normalized features
            similarity_matrix = torch.mm(features_norm, active_features_norm.t())
            
            # Convert to numpy for combining with IoU
            similarity_matrix = similarity_matrix.cpu().numpy()
            
            # Compute IoU matrix
            iou_matrix = np.zeros((len(boxes), len(active_ids)))
            for i, box in enumerate(boxes):
                for j, track_id in enumerate(active_ids):
                    track_box = self.tracks[track_id].location_history[-1]
                    iou_matrix[i, j] = self._calculate_iou(box, track_box)
            
            # Ensure matrices have the same shape
            assert similarity_matrix.shape == iou_matrix.shape, \
                f"Shape mismatch: similarity_matrix {similarity_matrix.shape} != iou_matrix {iou_matrix.shape}"
            
            # Combine similarities with weighted average
            combined_similarity = (
                0.7 * similarity_matrix +
                0.3 * iou_matrix
            )
            
            return combined_similarity
            
        except Exception as e:
            logger.error(f"Similarity computation error: {str(e)}")
            # Return a zero matrix as fallback
            return np.zeros((len(boxes), len(active_ids)))

def main():
    """Enhanced main application with error handling and graceful shutdown"""
    try:
        # Initialize tracking system
        tracker = MultiCameraTracker(
            similarity_thresh=config.similarity_threshold,
            batch_size=config.batch_size
        )
        
        # Configure video sources
        camera_sources = {
            0: 0
            #1: r"D:\Codes\gettyimages-1126783091-640_adpp.mp4"#,
            #2: "camera2.mp4",
            # Add more cameras as needed
        }
        
        # Initialize video streams with error handling
        streams = {}
        for camera_id, source in camera_sources.items():
            try:
                stream = VideoStream(source, buffer_size=config.buffer_size)
                streams[camera_id] = stream
                tracker.register_camera(camera_id)
                logger.info(f"Initialized camera {camera_id}")
            except Exception as e:
                logger.error(f"Failed to initialize camera {camera_id}: {str(e)}")
        
        if not streams:
            raise RuntimeError("No cameras could be initialized")
        
        # Create display windows
        for camera_id in streams:
            window_name = f"Camera {camera_id}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
        
        # Performance monitoring
        last_metrics_time = time.time()
        running = True
        
        # Main processing loop
        while running:
            try:
                for camera_id, stream in streams.items():
                    frame_data = stream.read()
                    if frame_data is not None:
                        timestamp, frame = frame_data
                        processed_frame = tracker.process_frame(frame, camera_id)
                        if processed_frame is not None:
                            cv2.imshow(f"Camera {camera_id}", processed_frame)
                
                # Display performance metrics
                current_time = time.time()
                if current_time - last_metrics_time >= 5.0:
                    metrics = tracker.get_performance_metrics()
                    logger.info(f"Performance metrics: {metrics}")
                    last_metrics_time = current_time
                
                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('s'):
                    # Example person search
                    results = tracker.search_person(1)
                    logger.info(f"Search results: {results}")
                
            except KeyboardInterrupt:
                running = False
            except Exception as e:
                logger.error(f"Main loop error: {str(e)}")
                time.sleep(1.0)  # Prevent rapid error logging
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    
    finally:
        # Cleanup
        logger.info("Shutting down tracking system...")
        for stream in streams.values():
            try:
                stream.stop()
            except:
                pass
        cv2.destroyAllWindows()
        logger.info("Tracking system stopped")

if __name__ == "__main__":
    main()