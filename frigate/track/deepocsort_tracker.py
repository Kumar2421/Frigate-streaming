"""DeepOCSORT tracker implementation with YOLO re-identification support."""

import logging
import random
import string
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from frigate.camera import PTZMetrics
from frigate.config import CameraConfig
from frigate.track import ObjectTracker
from frigate.util.image import SharedMemoryFrameManager

logger = logging.getLogger(__name__)

try:
    from deepocsort import DeepOCSORT
    from deepocsort.utils import get_embeddings
    DEEPOCSORT_AVAILABLE = True
except ImportError:
    logger.warning("DeepOCSORT not available. Install with: pip install deepocsort")
    DEEPOCSORT_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False


class ReIDExtractor:
    """Re-identification feature extractor using dedicated ReID models."""
    
    def __init__(self, model_path: str = "osnet_x1_0", device: str = "cpu"):
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard ReID input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            # Try to load torchreid models
            import torchreid
            self.model = torchreid.models.build_model(
                name=model_path,
                num_classes=1000,  # Dummy number of classes
                pretrained=True
            )
            self.model.to(device)
            self.model.eval()
            logger.info(f"ReID model loaded: {model_path}")
        except ImportError:
            logger.warning("torchreid not available. Install with: pip install torchreid")
            self._load_fallback_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load ReID model: {e}")
            self._load_fallback_model(model_path)
    
    def _load_fallback_model(self, model_path: str):
        """Load a fallback model if torchreid is not available."""
        try:
            # Try to load a pre-trained ResNet model as fallback
            import torchvision.models as models
            self.model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Fallback ResNet50 model loaded for ReID")
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            self.model = None
    
    def extract_features(self, image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract re-identification features from image region."""
        if self.model is None:
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            # Ensure bbox is within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.shape[1], int(x2))
            y2 = min(image.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            # Extract region of interest
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return None
                
            # Convert BGR to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            roi_tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(roi_tensor)
                # Flatten and normalize features
                features = features.view(features.size(0), -1)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                features = features.cpu().numpy().flatten()
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting ReID features: {e}")
            return None


class DeepOCSORTTracker(ObjectTracker):
    """DeepOCSORT tracker with ReID re-identification support."""
    
    def __init__(
        self,
        config: CameraConfig,
        ptz_metrics: PTZMetrics,
        reid_model_path: str = "osnet_x1_0",
        device: str = "cpu",
    ):
        if not DEEPOCSORT_AVAILABLE:
            raise ImportError("DeepOCSORT is not available. Please install it first.")
            
        self.camera_config = config
        self.detect_config = config.detect
        self.ptz_metrics = ptz_metrics
        self.camera_name = config.name
        self.frame_manager = SharedMemoryFrameManager()
        
        # Initialize DeepOCSORT tracker
        self.tracker = DeepOCSORT(
            det_thresh=0.3,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            delta_t=3,
            asso_func="giou",
            inertia=0.2,
            w_association_emb=0.75,
            alpha_fixed_emb=0.95,
            aw_param=0.5,
            embedding_off=False,
            cmc_off=False,
            aw_off=False,
            new_kf_off=False,
        )
        
        # Initialize ReID feature extractor
        self.reid_extractor = ReIDExtractor(reid_model_path, device)
        
        # Tracked objects storage
        self.tracked_objects: Dict[str, Dict[str, Any]] = {}
        self.track_id_map: Dict[str, str] = {}
        self.disappeared: Dict[str, int] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.stationary_box_history: Dict[str, List[List[int]]] = {}
        
        # Re-identification storage
        self.reid_features: Dict[str, np.ndarray] = {}
        self.reid_threshold = 0.7  # Similarity threshold for re-identification
        
        logger.info(f"DeepOCSORT tracker initialized for camera {self.camera_name}")
    
    def _generate_track_id(self, frame_time: float) -> str:
        """Generate a unique track ID."""
        rand_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{frame_time}-{rand_id}"
    
    def _extract_reid_features(self, frame: np.ndarray, detections: List[Tuple]) -> List[Optional[np.ndarray]]:
        """Extract re-identification features for all detections."""
        features = []
        for detection in detections:
            label, score, bbox, area, ratio, region = detection
            feature = self.reid_extractor.extract_features(frame, bbox)
            features.append(feature)
        return features
    
    def _compute_reid_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors."""
        if features1 is None or features2 is None:
            return 0.0
        return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    
    def _find_best_reid_match(self, new_features: np.ndarray, track_id: str) -> Optional[str]:
        """Find the best re-identification match for new features."""
        if new_features is None:
            return None
            
        best_match = None
        best_similarity = 0.0
        
        for existing_track_id, existing_features in self.reid_features.items():
            if existing_track_id == track_id:
                continue
                
            similarity = self._compute_reid_similarity(new_features, existing_features)
            if similarity > self.reid_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_track_id
                
        return best_match if best_similarity > self.reid_threshold else None
    
    def register(self, track_id: str, obj: Dict[str, Any]) -> None:
        """Register a new tracked object."""
        frigate_id = self._generate_track_id(obj["frame_time"])
        self.track_id_map[track_id] = frigate_id
        obj["id"] = frigate_id
        obj["start_time"] = obj["frame_time"]
        obj["motionless_count"] = 0
        obj["position_changes"] = 0
        obj["reid_matches"] = []  # Track re-identification matches
        
        self.tracked_objects[frigate_id] = obj
        self.disappeared[frigate_id] = 0
        
        # Initialize position tracking
        box = obj["box"]
        self.positions[frigate_id] = {
            "xmins": [box[0]],
            "ymins": [box[1]],
            "xmaxs": [box[2]],
            "ymaxs": [box[3]],
            "xmin": box[0],
            "ymin": box[1],
            "xmax": box[2],
            "ymax": box[3],
        }
        self.stationary_box_history[frigate_id] = [box]
        
        logger.debug(f"Registered new object {frigate_id} with track_id {track_id}")
    
    def deregister(self, frigate_id: str, track_id: str) -> None:
        """Deregister a tracked object."""
        if frigate_id in self.tracked_objects:
            del self.tracked_objects[frigate_id]
        if frigate_id in self.disappeared:
            del self.disappeared[frigate_id]
        if frigate_id in self.positions:
            del self.positions[frigate_id]
        if frigate_id in self.stationary_box_history:
            del self.stationary_box_history[frigate_id]
        if frigate_id in self.reid_features:
            del self.reid_features[frigate_id]
        if track_id in self.track_id_map:
            del self.track_id_map[track_id]
            
        logger.debug(f"Deregistered object {frigate_id} with track_id {track_id}")
    
    def update(self, track_id: str, obj: Dict[str, Any]) -> None:
        """Update an existing tracked object."""
        frigate_id = self.track_id_map.get(track_id)
        if frigate_id is None:
            return
            
        self.disappeared[frigate_id] = 0
        
        # Update position tracking
        box = obj["box"]
        self.stationary_box_history[frigate_id].append(box)
        if len(self.stationary_box_history[frigate_id]) > 10:
            self.stationary_box_history[frigate_id] = self.stationary_box_history[frigate_id][-10:]
        
        # Update object data
        self.tracked_objects[frigate_id].update(obj)
    
    def match_and_update(
        self,
        frame_name: str,
        frame_time: float,
        detections: List[Tuple[Any, Any, Any, Any, Any, Any]],
    ) -> None:
        """Main tracking update method."""
        if not detections:
            return
            
        # Get frame for re-identification
        frame = None
        if self.ptz_metrics.autotracker_enabled.value:
            frame = self.frame_manager.get(frame_name, self.camera_config.frame_shape_yuv)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
        
        # Extract re-identification features
        reid_features = []
        if frame is not None:
            reid_features = self._extract_reid_features(frame, detections)
        
        # Convert detections to DeepOCSORT format
        deepocsort_detections = []
        for i, detection in enumerate(detections):
            label, score, bbox, area, ratio, region = detection
            x1, y1, x2, y2 = bbox
            
            # Convert to DeepOCSORT format [x1, y1, x2, y2, score, class_id]
            class_id = 0  # Default class ID, could be mapped from label
            deepocsort_det = [x1, y1, x2, y2, score, class_id]
            
            # Add re-identification features if available
            if i < len(reid_features) and reid_features[i] is not None:
                deepocsort_det.append(reid_features[i])
            
            deepocsort_detections.append(deepocsort_det)
        
        # Update DeepOCSORT tracker
        if deepocsort_detections:
            tracked_objects = self.tracker.update(
                np.array(deepocsort_detections), 
                frame if frame is not None else None
            )
        else:
            tracked_objects = self.tracker.update(np.array([]), None)
        
        # Process tracked objects
        active_track_ids = set()
        for track in tracked_objects:
            track_id = str(track[4])  # DeepOCSORT track ID
            x1, y1, x2, y2, score = track[:5]
            
            # Convert back to Frigate format
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            area = (x2 - x1) * (y2 - y1)
            ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
            
            obj_data = {
                "label": "person",  # Default label, could be determined from detection
                "score": float(score),
                "box": bbox,
                "area": int(area),
                "ratio": ratio,
                "region": bbox,  # Same as box for now
                "frame_time": frame_time,
            }
            
            active_track_ids.add(track_id)
            
            # Check for re-identification matches
            if len(track) > 5:  # Has re-identification features
                new_features = track[5]
                if new_features is not None:
                    # Store features for this track
                    self.reid_features[track_id] = new_features
                    
                    # Check for matches with existing tracks
                    best_match = self._find_best_reid_match(new_features, track_id)
                    if best_match:
                        # Record re-identification match
                        if track_id not in self.track_id_map:
                            self.register(track_id, obj_data)
                        
                        frigate_id = self.track_id_map[track_id]
                        if "reid_matches" not in self.tracked_objects[frigate_id]:
                            self.tracked_objects[frigate_id]["reid_matches"] = []
                        
                        match_info = {
                            "matched_track_id": best_match,
                            "similarity": self._compute_reid_similarity(new_features, self.reid_features[best_match]),
                            "timestamp": frame_time
                        }
                        self.tracked_objects[frigate_id]["reid_matches"].append(match_info)
                        
                        logger.info(f"Re-identification match: {track_id} -> {best_match} (similarity: {match_info['similarity']:.3f})")
            
            # Register or update object
            if track_id not in self.track_id_map:
                self.register(track_id, obj_data)
            else:
                self.update(track_id, obj_data)
        
        # Clean up disappeared tracks
        disappeared_tracks = set(self.track_id_map.keys()) - active_track_ids
        for track_id in disappeared_tracks:
            frigate_id = self.track_id_map[track_id]
            self.disappeared[frigate_id] += 1
            
            # Remove tracks that have been missing for too long
            if self.disappeared[frigate_id] > 30:  # 30 frames threshold
                self.deregister(frigate_id, track_id)
    
    def get_tracked_objects(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently tracked objects."""
        return self.tracked_objects
    
    def get_reid_matches(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get re-identification matches for all tracked objects."""
        reid_matches = {}
        for frigate_id, obj in self.tracked_objects.items():
            if "reid_matches" in obj and obj["reid_matches"]:
                reid_matches[frigate_id] = obj["reid_matches"]
        return reid_matches
