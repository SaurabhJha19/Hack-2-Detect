import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models.resnet import ResNet18_Weights
from facenet_pytorch import MTCNN, InceptionResnetV1
from collections import deque
import time
import warnings
from threading import Thread, Lock
from queue import Queue
import sqlite3
from datetime import datetime
import logging
from typing import Optional, Dict, List

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings and configure device
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
half_precision = torch.cuda.is_available()

class VideoStream:
    """Handles video capture from various sources (webcam, IP camera, video file)"""
    def __init__(self, source: any, buffer_size: int = 30):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.queue = Queue(maxsize=buffer_size)
        self.lock = Lock()
        self.running = True
        
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Wait for camera initialization
        time.sleep(1.0)
    
    def _update(self):
        while self.running:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (1280, 720))
                    with self.lock:
                        self.queue.put((time.time(), frame))
                else:
                    if isinstance(self.source, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        break
            else:
                time.sleep(0.001)
    
    def read(self):
        return self.queue.get() if not self.queue.empty() else None
    
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

class DatabaseManager:
    """Manages persistent storage of tracking data"""
    def __init__(self, db_path: str = 'tracking_database.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup_database()
    
    def setup_database(self):
        self.cursor.executescript('''
            CREATE TABLE IF NOT EXISTS tracked_persons (
                person_id INTEGER PRIMARY KEY,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                feature_vector BLOB
            );
            
            CREATE TABLE IF NOT EXISTS location_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                camera_id INTEGER,
                timestamp TIMESTAMP,
                x1 FLOAT,
                y1 FLOAT,
                x2 FLOAT,
                y2 FLOAT,
                confidence FLOAT,
                FOREIGN KEY (person_id) REFERENCES tracked_persons(person_id)
            );
        ''')
        self.conn.commit()
    
    def add_person(self, features: np.ndarray) -> int:
        self.cursor.execute('''
            INSERT INTO tracked_persons (first_seen, last_seen, feature_vector)
            VALUES (?, ?, ?)
        ''', (datetime.now(), datetime.now(), features.tobytes()))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def update_location(self, person_id: int, camera_id: int, 
                       box: np.ndarray, confidence: float):
        self.cursor.execute('''
            INSERT INTO location_history 
            (person_id, camera_id, timestamp, x1, y1, x2, y2, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (person_id, camera_id, datetime.now(), 
              float(box[0]), float(box[1]), float(box[2]), float(box[3]),
              float(confidence)))
        self.conn.commit()

class FeatureExtractor:
    """Extracts facial and clothing features from frames"""
    def __init__(self):
        # Initialize face detection and recognition models
        self.face_detector = MTCNN(
            keep_all=True,
            device=device,
            thresholds=[0.7, 0.8, 0.9],
            min_face_size=40
        ).eval()
        
        self.face_recognizer = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'
        ).eval().to(device)
        
        self.clothing_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.clothing_model.fc = torch.nn.Identity()
        self.clothing_model = self.clothing_model.eval().to(device)
        
        if half_precision:
            self.face_recognizer = self.face_recognizer.half()
            self.clothing_model = self.clothing_model.half()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Feature dimensions
        self.face_dim = 512
        self.clothing_dim = 512
        self.motion_dim = 64
        self.total_dim = self.face_dim + self.clothing_dim + self.motion_dim
        
        # Motion tracking
        self.prev_frame = None
        logger.info(f"Initialized FeatureExtractor with total dimension: {self.total_dim}")
    
    def get_features(self, frame: np.ndarray, boxes: np.ndarray) -> Optional[torch.Tensor]:
        with torch.no_grad():
            # Get individual features
            face_features = self.get_face_features(frame, boxes)
            clothing_features = self.get_clothing_features(frame, boxes)
            motion_features = self.get_motion_features(frame, boxes)
            
            # Check if any features are missing
            if face_features is None or clothing_features is None:
                return None
            
            # Convert motion features to tensor
            motion_tensor = torch.from_numpy(motion_features).float().to(device)
            
            # Combine all features
            combined = torch.cat([
                face_features,
                clothing_features,
                motion_tensor
            ], dim=1)
            
            # Validate dimensions
            if combined.size(1) != self.total_dim:
                logger.error(f"Feature dimension mismatch. Expected {self.total_dim}, got {combined.size(1)}")
                return None
            
            return combined
    
    def get_face_features(self, frame: np.ndarray, boxes: np.ndarray) -> Optional[torch.Tensor]:
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                face = cv2.resize(face, (160, 160))
                face = torch.tensor(face).permute(2, 0, 1).float()
                if half_precision:
                    face = face.half()
                face = face.to(device)
                faces.append(face)
        
        if not faces:
            return None
            
        face_features = self.face_recognizer(torch.stack(faces))
        return face_features
    
    def get_clothing_features(self, frame: np.ndarray, boxes: np.ndarray) -> Optional[torch.Tensor]:
        features = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            y2 = min(int(y2 + (y2 - y1) * 0.5), frame.shape[0])
            clothing = frame[y1:y2, x1:x2]
            if clothing.size > 0:
                tensor = self.transform(clothing).unsqueeze(0)
                if half_precision:
                    tensor = tensor.half()
                tensor = tensor.to(device)
                features.append(self.clothing_model(tensor))
        
        return torch.cat(features, dim=0) if features else None
    
    def get_motion_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return np.zeros((len(boxes), self.motion_dim))
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, current_frame,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        motion_features = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            roi_flow = flow[y1:y2, x1:x2]
            if roi_flow.size > 0:
                # Calculate fixed-size motion histogram
                hist = np.histogram2d(
                    roi_flow[..., 0].flatten(),
                    roi_flow[..., 1].flatten(),
                    bins=8
                )[0].flatten()
                
                # Ensure correct dimension
                if len(hist) < self.motion_dim:
                    hist = np.pad(hist, (0, self.motion_dim - len(hist)))
                elif len(hist) > self.motion_dim:
                    hist = hist[:self.motion_dim]
                    
                motion_features.append(hist)
            else:
                motion_features.append(np.zeros(self.motion_dim))
        
        self.prev_frame = current_frame
        return np.array(motion_features)

class TrackedPerson:
    """Represents a tracked individual across cameras"""
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
        self.update_location(box, camera_id)
    
    def update_features(self, features: torch.Tensor):
        self.feature_queue.append(features)
        self.last_seen = time.time()
    
    def update_location(self, box: np.ndarray, camera_id: int):
        self.location_history.append(np.array(box, dtype=float))
        self.camera_history.append(camera_id)
        self.last_camera_id = camera_id
    
    def get_avg_features(self) -> torch.Tensor:
        return torch.mean(torch.stack(list(self.feature_queue)), dim=0)

class MultiCameraTracker:
    """Main tracking system that manages multiple camera feeds"""
    def __init__(self, similarity_thresh: float = 0.85):
        self.feature_extractor = FeatureExtractor()
        self.tracks: Dict[int, TrackedPerson] = {}
        self.db_manager = DatabaseManager()
        self.lock = Lock()
        self.similarity_thresh = similarity_thresh
        self.camera_buffers: Dict[int, deque] = {}
        self.max_cosine_sim = torch.nn.CosineSimilarity(dim=1)
    
    def register_camera(self, camera_id: int):
        self.camera_buffers[camera_id] = deque(maxlen=300)
        logger.info(f"Registered camera {camera_id}")
    
    def process_frame(self, frame: np.ndarray, camera_id: int) -> Optional[np.ndarray]:
        if frame is None:
            return None
        
        # Store frame in buffer
        self.camera_buffers[camera_id].append({
            'frame': frame.copy(),
            'timestamp': time.time()
        })
        
        # Detect faces
        boxes, _ = self.feature_extractor.face_detector.detect(frame)
        if boxes is None:
            return frame
        
        # Extract features
        features = self.feature_extractor.get_features(frame, boxes)
        if features is None:
            return frame
        
        # Update tracking
        with self.lock:
            self.update_tracks(features, boxes, camera_id)
            self.clean_old_tracks()
        
        return self.draw_tracking(frame, boxes, camera_id)
    
    def update_tracks(self, features: torch.Tensor, boxes: np.ndarray, camera_id: int):
        active_ids = list(self.tracks.keys())
        
        if active_ids:
            # Get features for existing tracks
            active_features = torch.stack([t.get_avg_features() for t in self.tracks.values()])
            
            # Verify dimensions match
            if active_features.size(1) != features.size(1):
                logger.error(f"Feature dimension mismatch: active={active_features.size(1)}, new={features.size(1)}")
                return
            
            # Calculate similarities
            similarities = self.max_cosine_sim(
                features.unsqueeze(1),
                active_features.unsqueeze(0)
            )
            
            # Match detections to tracks
            matched_pairs = {}
            for i, sims in enumerate(similarities):
                max_val, max_idx = torch.max(sims, dim=0)
                if max_val > self.similarity_thresh:
                    track_id = active_ids[max_idx]
                    matched_pairs[track_id] = i
                    
                    # Update existing track
                    self.tracks[track_id].update_features(features[i])
                    self.tracks[track_id].update_location(boxes[i], camera_id)
                    self.db_manager.update_location(track_id, camera_id, boxes[i], float(max_val))
        else:
            matched_pairs = {}
        
        # Create new tracks for unmatched detections
        for i in range(len(features)):
            if i not in matched_pairs.values():
                # Add to database and create new track
                person_id = self.db_manager.add_person(features[i].cpu().numpy())
                new_track = TrackedPerson(person_id, features[i], boxes[i], camera_id)
                self.tracks[person_id] = new_track
                
                # Update initial location
                self.db_manager.update_location(
                    person_id, camera_id, boxes[i], 1.0  # New tracks start with confidence 1.0
                )
    
    def clean_old_tracks(self, max_age: float = 15.0):
        """Remove tracks that haven't been seen for a while"""
        current_time = time.time()
        to_delete = [
            tid for tid, track in self.tracks.items()
            if current_time - track.last_seen > max_age
        ]
        for tid in to_delete:
            del self.tracks[tid]
    
    def draw_tracking(self, frame: np.ndarray, boxes: np.ndarray, 
                     camera_id: int) -> np.ndarray:
        """Draw tracking visualization on frame"""
        # Create a copy of the frame for drawing
        vis_frame = frame.copy()
        
        # Draw camera ID
        cv2.putText(vis_frame, f"Camera {camera_id}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            track_id = None
            color = (0, 255, 0)  # Default color for unmatched detections
            
            # Find matching track
            with self.lock:
                for tid, track in self.tracks.items():
                    if np.allclose(np.array(box, dtype=float), 
                                 track.location_history[-1]):
                        track_id = tid
                        color = track.color
                        break
            
            if track_id is not None:
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and camera transition info
                track = self.tracks[track_id]
                label = f"ID: {track_id}"
                if track.last_camera_id != camera_id:
                    label += f" (From Cam {track.last_camera_id})"
                
                cv2.putText(vis_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw movement trail
                points = []
                for loc in track.location_history:
                    cx = int((loc[0] + loc[2]) / 2)
                    cy = int((loc[1] + loc[3]) / 2)
                    points.append((cx, cy))
                
                if len(points) > 1:
                    points = np.array(points, np.int32)
                    cv2.polylines(vis_frame, [points], False, color, 2)
        
        return vis_frame
    
    def search_person(self, target_id: int, time_window: float = 300) -> List[Dict]:
        """Search for a person across all camera feeds"""
        results = []
        current_time = time.time()
        
        with self.lock:
            # Check if person is currently being tracked
            if target_id in self.tracks:
                target_features = self.tracks[target_id].get_avg_features()
                
                # Search through recent frames in all cameras
                for camera_id, buffer in self.camera_buffers.items():
                    for frame_data in buffer:
                        if current_time - frame_data['timestamp'] > time_window:
                            continue
                        
                        frame = frame_data['frame']
                        boxes, _ = self.feature_extractor.face_detector.detect(frame)
                        
                        if boxes is not None:
                            features = self.feature_extractor.get_features(frame, boxes)
                            if features is not None:
                                similarities = self.max_cosine_sim(
                                    features,
                                    target_features.unsqueeze(0)
                                )
                                
                                max_sim, max_idx = torch.max(similarities, dim=0)
                                if max_sim > self.similarity_thresh:
                                    results.append({
                                        'camera_id': camera_id,
                                        'timestamp': frame_data['timestamp'],
                                        'confidence': float(max_sim),
                                        'location': boxes[max_idx].tolist()
                                    })
        
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)

def main():
    """Main function to run the tracking system"""
    # Initialize the tracking system
    tracker = MultiCameraTracker(similarity_thresh=0.85)
    
    # Create video streams for each camera
    streams = {}
    camera_sources = {
        0: 0#,  # Local webcam
        #1: "videos/camera1.mp4",  # Video file
        #2: "rtsp://camera2_url"  # IP camera
    }
    
    # Initialize video streams
    for camera_id, source in camera_sources.items():
        try:
            stream = VideoStream(source)
            streams[camera_id] = stream
            tracker.register_camera(camera_id)
            logger.info(f"Initialized camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to initialize camera {camera_id}: {str(e)}")
    
    # Create windows for each camera
    for camera_id in streams.keys():
        window_name = f"Camera {camera_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    try:
        while True:
            # Process each camera
            for camera_id, stream in streams.items():
                frame_data = stream.read()
                if frame_data is not None:
                    timestamp, frame = frame_data
                    processed_frame = tracker.process_frame(frame, camera_id)
                    if processed_frame is not None:
                        cv2.imshow(f"Camera {camera_id}", processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Example: Search for person with ID 1
                results = tracker.search_person(1)
                logger.info(f"Search results: {results}")
    
    except KeyboardInterrupt:
        logger.info("Stopping tracking system...")
    
    finally:
        # Clean up
        for stream in streams.values():
            stream.stop()
        cv2.destroyAllWindows()
        logger.info("Tracking system stopped")

if __name__ == "__main__":
    main()