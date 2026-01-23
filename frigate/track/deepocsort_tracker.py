"""DeepOCSORT tracker implementation with YOLO re-identification support."""

import logging
import os
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
from frigate.util.image import intersection_over_union
from frigate.util.object import median_of_boxes

logger = logging.getLogger(__name__)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_REALTIME_AVAILABLE = True
except ImportError:
    logger.warning(
        "DeepSort Realtime not available. Install with: pip install deep-sort-realtime"
    )
    DEEPSORT_REALTIME_AVAILABLE = False

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

            # If a .pth file is provided, load it into an OSNet backbone.
            # This supports configs like /config/*.pth mounted into the container.
            if model_path.endswith(".pth") and os.path.exists(model_path):
                backbone = "osnet_x1_0"
                self.model = torchreid.models.build_model(
                    name=backbone,
                    num_classes=1000,  # Dummy number of classes
                    pretrained=False,
                )
                state = torch.load(model_path, map_location=device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if isinstance(state, dict):
                    cleaned = {}
                    for k, v in state.items():
                        nk = k
                        if nk.startswith("module."):
                            nk = nk[len("module.") :]
                        cleaned[nk] = v
                    self.model.load_state_dict(cleaned, strict=False)
                self.model.to(device)
                self.model.eval()
                logger.info(f"ReID weights loaded from file: {model_path}")
            else:
                self.model = torchreid.models.build_model(
                    name=model_path,
                    num_classes=1000,  # Dummy number of classes
                    pretrained=False,
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

            try:
                self.model = models.resnet50(weights=None)
            except TypeError:
                # older torchvision
                self.model = models.resnet50(pretrained=False)
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
    """DeepSort Realtime tracker with ReID re-identification support."""

    def _normalize_detection(
        self, detection: Any
    ) -> Optional[Tuple[str, float, List[int], float, float, List[int]]]:
        """Normalize input detection into (label, score, bbox, area, ratio, region).

        Supports:
        - tuples/lists from Frigate internal pipelines
        - dict detections (e.g. {label, score, box})
        - objects with .obj_data dict
        """
        try:
            if isinstance(detection, (tuple, list)):
                # Some pipelines may pass extra fields. We only need the first 6.
                if len(detection) < 3:
                    return None
                label = detection[0]
                score = float(detection[1])
                bbox = detection[2]
                area = detection[3] if len(detection) > 3 else None
                ratio = detection[4] if len(detection) > 4 else None
                region = detection[5] if len(detection) > 5 else None

                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    return None
                x1, y1, x2, y2 = [int(v) for v in bbox]
                if area is None:
                    area = float(max(0, x2 - x1) * max(0, y2 - y1))
                if ratio is None:
                    ratio = float((x2 - x1) / (y2 - y1)) if (y2 - y1) > 0 else 1.0
                if region is None:
                    region = [x1, y1, x2, y2]
                return (str(label), float(score), [x1, y1, x2, y2], float(area), float(ratio), list(region))

            if hasattr(detection, "obj_data") and isinstance(getattr(detection, "obj_data"), dict):
                detection = detection.obj_data

            if isinstance(detection, dict):
                label = detection.get("label") or detection.get("name")
                if label is None:
                    return None
                score = float(detection.get("score", 1.0) or 1.0)
                bbox = detection.get("box") or detection.get("bbox") or detection.get("region")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    return None
                x1, y1, x2, y2 = [int(v) for v in bbox]
                area = float(detection.get("area") or (max(0, x2 - x1) * max(0, y2 - y1)))
                ratio = float(detection.get("ratio") or (((x2 - x1) / (y2 - y1)) if (y2 - y1) > 0 else 1.0))
                region = detection.get("region")
                if not isinstance(region, (list, tuple)) or len(region) != 4:
                    region = [x1, y1, x2, y2]
                return (str(label), float(score), [x1, y1, x2, y2], float(area), float(ratio), list(region))
        except Exception:
            return None

        return None
    
    def __init__(
        self,
        config: CameraConfig,
        ptz_metrics: PTZMetrics,
        reid_model_path: str = "osnet_x1_0",
        device: str = "cpu",
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        reid_threshold: float = 0.7,
        use_track_id_as_reid: bool = False,
    ):
        if not DEEPSORT_REALTIME_AVAILABLE:
            raise ImportError(
                "DeepSort Realtime is not available. Please install it first."
            )
            
        self.camera_config = config
        self.detect_config = config.detect
        self.ptz_metrics = ptz_metrics
        self.camera_name = config.name
        self.frame_manager = SharedMemoryFrameManager()
        
        self.max_age = int(max_age)
        self.use_track_id_as_reid = bool(use_track_id_as_reid)

        # Initialize DeepSort Realtime tracker
        # We provide `frame=` each call, so it can compute its internal embeddings if desired.
        self.tracker = DeepSort(
            max_age=int(max_age),
            n_init=int(n_init),
            max_iou_distance=float(max_iou_distance),
            max_cosine_distance=float(max_cosine_distance),
            nn_budget=int(nn_budget),
            embedder=None,
            half=False,
            bgr=True,
            embedder_gpu=False,
            polygon=False,
        )
        
        # Initialize ReID feature extractor
        self.reid_extractor = ReIDExtractor(reid_model_path, device)

        # Cache embedding dimension so we can safely fill missing embeddings.
        # deep-sort-realtime expects a list of fixed-length vectors.
        self._embed_dim: Optional[int] = None
        
        # Tracked objects storage
        self.tracked_objects: Dict[str, Dict[str, Any]] = {}
        self.track_id_map: Dict[str, str] = {}
        self.disappeared: Dict[str, int] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.stationary_box_history: Dict[str, List[List[int]]] = {}
        # Boxes which don't yet have a confirmed tracked object (used by Frigate region consolidation)
        self.untracked_object_boxes: List[List[int]] = []
        
        # Re-identification storage
        # Track-level embeddings for debugging / matching continuity.
        self.track_features: Dict[str, np.ndarray] = {}
        # Persistent identity store (stable IDs within runtime).
        self.identity_features: Dict[str, np.ndarray] = {}
        self.track_identity_map: Dict[str, str] = {}
        self._next_identity_id: int = 1
        self.reid_threshold = reid_threshold  # Similarity threshold for re-identification
        
        logger.info(f"DeepSort Realtime tracker initialized for camera {self.camera_name}")
    
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
    
    def _find_best_identity_match(self, new_features: np.ndarray) -> Optional[Tuple[str, float]]:
        """Find the best existing identity match for new features."""
        if new_features is None:
            return None

        best_identity_id: str | None = None
        best_similarity = 0.0

        for identity_id, identity_features in self.identity_features.items():
            similarity = self._compute_reid_similarity(new_features, identity_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_identity_id = identity_id

        if best_identity_id is None or best_similarity < self.reid_threshold:
            return None

        return (best_identity_id, best_similarity)

    def _assign_identity(self, track_id: str, new_features: Optional[np.ndarray]) -> Optional[str]:
        if track_id in self.track_identity_map:
            identity_id = self.track_identity_map[track_id]
            if new_features is not None:
                # EMA update of identity prototype
                prev = self.identity_features.get(identity_id)
                if prev is None:
                    self.identity_features[identity_id] = new_features
                else:
                    updated = (0.9 * prev) + (0.1 * new_features)
                    # normalize
                    denom = float(np.linalg.norm(updated))
                    if denom > 0:
                        updated = updated / denom
                    self.identity_features[identity_id] = updated
            return identity_id

        if new_features is None:
            return None

        match = self._find_best_identity_match(new_features)
        if match is not None:
            identity_id, similarity = match
            self.track_identity_map[track_id] = identity_id
            # update identity prototype
            prev = self.identity_features.get(identity_id)
            if prev is None:
                self.identity_features[identity_id] = new_features
            else:
                updated = (0.9 * prev) + (0.1 * new_features)
                denom = float(np.linalg.norm(updated))
                if denom > 0:
                    updated = updated / denom
                self.identity_features[identity_id] = updated
            return identity_id

        # New identity
        identity_id = f"reid-{self._next_identity_id}"
        self._next_identity_id += 1
        self.identity_features[identity_id] = new_features
        self.track_identity_map[track_id] = identity_id
        return identity_id
    
    def register(self, track_id: str, obj: Dict[str, Any]) -> None:
        """Register a new tracked object."""
        frigate_id = self._generate_track_id(obj["frame_time"])
        self.track_id_map[track_id] = frigate_id
        obj["id"] = frigate_id
        obj["start_time"] = obj["frame_time"]
        obj["motionless_count"] = 0
        obj["position_changes"] = 0
        # Frigate expects score history to exist and be a list
        obj["score_history"] = [float(obj.get("score", 0.0))]
        # Frigate expects centroid/estimate fields
        try:
            x1, y1, x2, y2 = obj["box"]
            obj["centroid"] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
        except Exception:
            obj["centroid"] = (0, 0)
        obj["estimate"] = obj.get("box")
        obj["estimate_velocity"] = (0, 0)
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
        
        # Object registered successfully

    def _update_position(self, frigate_id: str, box: List[int], stationary: bool) -> bool:
        position = self.positions.get(frigate_id)
        if position is None:
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
            return False

        position_box = (
            int(position.get("xmin", box[0])),
            int(position.get("ymin", box[1])),
            int(position.get("xmax", box[2])),
            int(position.get("ymax", box[3])),
        )
        b = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

        threshold = 0.9 if stationary else 0.6

        hist = self.stationary_box_history.get(frigate_id)
        if hist is None:
            hist = []
            self.stationary_box_history[frigate_id] = hist
        hist.append([int(v) for v in box])
        if len(hist) > 10:
            self.stationary_box_history[frigate_id] = hist[-10:]
            hist = self.stationary_box_history[frigate_id]

        try:
            median_iou = intersection_over_union(position_box, median_of_boxes(hist))
        except Exception:
            median_iou = intersection_over_union(position_box, b)

        if median_iou < threshold:
            self.positions[frigate_id] = {
                "xmins": [b[0]],
                "ymins": [b[1]],
                "xmaxs": [b[2]],
                "ymaxs": [b[3]],
                "xmin": b[0],
                "ymin": b[1],
                "xmax": b[2],
                "ymax": b[3],
            }
            self.stationary_box_history[frigate_id] = [list(b)]
            return False

        try:
            if 5 <= len(position.get("xmins", [])) < 10:
                position["xmins"].append(b[0])
                position["ymins"].append(b[1])
                position["xmaxs"].append(b[2])
                position["ymaxs"].append(b[3])
                position["xmin"] = float(np.percentile(position["xmins"], 15))
                position["ymin"] = float(np.percentile(position["ymins"], 15))
                position["xmax"] = float(np.percentile(position["xmaxs"], 85))
                position["ymax"] = float(np.percentile(position["ymaxs"], 85))
        except Exception:
            pass

        return True

    def _is_expired(self, frigate_id: str) -> bool:
        obj = self.tracked_objects.get(frigate_id)
        if obj is None:
            return False

        try:
            max_frames = self.detect_config.stationary.max_frames.objects.get(
                obj["label"],
                self.detect_config.stationary.max_frames.default,
            )
        except Exception:
            return False

        if max_frames is None:
            return False

        try:
            return (
                int(obj.get("motionless_count", 0))
                - int(self.detect_config.stationary.threshold)
                > int(max_frames)
            )
        except Exception:
            return False
    
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
        # track_id is the DeepOCSORT track id (string)
        if track_id in self.track_features:
            del self.track_features[track_id]
        if track_id in self.track_identity_map:
            # Keep identity_features for persistence within runtime.
            del self.track_identity_map[track_id]
        if track_id in self.track_id_map:
            del self.track_id_map[track_id]
            
        # Object deregistered successfully
    
    def update(self, track_id: str, obj: Dict[str, Any]) -> None:
        """Update an existing tracked object."""
        frigate_id = self.track_id_map.get(track_id)
        if frigate_id is None:
            return
            
        self.disappeared[frigate_id] = 0

        stationary = False
        try:
            stationary = (
                int(self.tracked_objects[frigate_id].get("motionless_count", 0))
                >= int(self.detect_config.stationary.threshold)
            )
        except Exception:
            stationary = False

        moved_within_position = True
        try:
            moved_within_position = self._update_position(
                frigate_id,
                [int(v) for v in obj.get("box", self.tracked_objects[frigate_id].get("box", [0, 0, 0, 0]))],
                stationary,
            )
        except Exception:
            moved_within_position = True

        if moved_within_position:
            try:
                self.tracked_objects[frigate_id]["motionless_count"] = int(
                    self.tracked_objects[frigate_id].get("motionless_count", 0)
                ) + 1
            except Exception:
                self.tracked_objects[frigate_id]["motionless_count"] = 1
            if self._is_expired(frigate_id):
                self.deregister(frigate_id, track_id)
                return
        else:
            try:
                if (
                    int(self.tracked_objects[frigate_id].get("position_changes", 0)) == 0
                    or int(self.tracked_objects[frigate_id].get("motionless_count", 0))
                    >= int(self.detect_config.stationary.threshold)
                ):
                    self.tracked_objects[frigate_id]["position_changes"] = int(
                        self.tracked_objects[frigate_id].get("position_changes", 0)
                    ) + 1
            except Exception:
                self.tracked_objects[frigate_id]["position_changes"] = int(
                    self.tracked_objects[frigate_id].get("position_changes", 0)
                ) + 1

            self.tracked_objects[frigate_id]["motionless_count"] = 0

        # Maintain Frigate-required fields
        try:
            x1, y1, x2, y2 = obj["box"]
            obj["centroid"] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
        except Exception:
            pass
        obj["estimate"] = obj.get("box")
        if "estimate_velocity" not in self.tracked_objects[frigate_id]:
            self.tracked_objects[frigate_id]["estimate_velocity"] = (0, 0)

        # score_history: append current score and keep last 10
        try:
            hist = self.tracked_objects[frigate_id].get("score_history")
            if not isinstance(hist, list):
                hist = []
            hist.append(float(obj.get("score", 0.0)))
            if len(hist) > 10:
                hist = hist[-10:]
            obj["score_history"] = hist
        except Exception:
            obj["score_history"] = self.tracked_objects[frigate_id].get("score_history", [0.0])

        # Preserve motionless_count/position_changes accumulated by Frigate logic
        for k in ("motionless_count", "position_changes", "start_time"):
            if k in self.tracked_objects[frigate_id] and k not in obj:
                obj[k] = self.tracked_objects[frigate_id][k]

        # Update object data
        self.tracked_objects[frigate_id].update(obj)
    
    def match_and_update(
        self,
        frame_name: str,
        frame_time: float,
        detections: List[Tuple[Any, Any, Any, Any, Any, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Main tracking update method."""
        if not detections:
            # Still allow internal cleanup via disappeared bookkeeping.
            detections = []
            
        # Get frame for re-identification
        bgr_frame = None
        if frame is not None:
            # `frame` from process_frames is usually YUV I420.
            try:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            except Exception:
                # best effort if already BGR
                bgr_frame = frame
        elif self.ptz_metrics.autotracker_enabled.value:
            bgr_frame = self.frame_manager.get(frame_name, self.camera_config.frame_shape_yuv)
            if bgr_frame is not None:
                try:
                    bgr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_YUV2BGR_I420)
                except Exception:
                    pass

        # Convert detections to deep_sort_realtime format:
        # ([left, top, w, h], confidence, detection_class)
        #
        # IMPORTANT: We initialize DeepSort with embedder=None to avoid implicit weight downloads.
        # In that mode deep-sort-realtime REQUIRES embeddings to be passed on each update.
        bbs = []
        embeds = []
        for detection in detections:
            norm = self._normalize_detection(detection)
            if norm is None:
                continue
            label, score, bbox, area, ratio, region = norm
            x1, y1, x2, y2 = bbox
            bbs.append(
                ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(score), str(label))
            )

            feat = None
            if bgr_frame is not None:
                try:
                    feat = self.reid_extractor.extract_features(bgr_frame, [int(x1), int(y1), int(x2), int(y2)])
                except Exception:
                    feat = None
            if feat is not None:
                try:
                    feat_arr = np.asarray(feat, dtype=np.float32)
                    feat_arr = feat_arr.reshape(-1)
                    if feat_arr.size > 0:
                        if self._embed_dim is None:
                            self._embed_dim = int(feat_arr.size)
                        elif int(feat_arr.size) != int(self._embed_dim):
                            feat_arr = None
                    else:
                        feat_arr = None
                except Exception:
                    feat_arr = None
            else:
                feat_arr = None

            if feat_arr is None:
                dim = int(self._embed_dim or 128)
                feat_arr = np.zeros((dim,), dtype=np.float32)
                if self._embed_dim is None:
                    self._embed_dim = dim

            # Ensure embeddings are always finite and have non-zero norm.
            # deep-sort-realtime normalizes embeddings internally and will produce NaNs
            # if the vector has zero norm or contains NaN/Inf values.
            try:
                feat_arr = np.asarray(feat_arr, dtype=np.float32).reshape(-1)
                feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
                nrm = float(np.linalg.norm(feat_arr))
                if not np.isfinite(nrm) or nrm < 1e-6:
                    # Set a deterministic non-zero vector (prevents divide-by-zero)
                    feat_arr[:] = 0.0
                    if feat_arr.size > 0:
                        feat_arr[0] = 1.0
            except Exception:
                dim = int(self._embed_dim or 128)
                feat_arr = np.zeros((dim,), dtype=np.float32)
                feat_arr[0] = 1.0
            embeds.append(feat_arr)

        # If the embedder is not configured, deep-sort-realtime requires embeds.
        # Provide them even if some are None.
        if not bbs:
            tracks = self.tracker.update_tracks([], embeds=[], frame=bgr_frame)
        else:
            tracks = self.tracker.update_tracks(bbs, embeds=embeds, frame=bgr_frame)

        active_track_ids: set[str] = set()
        for track in tracks:
            try:
                if hasattr(track, "is_confirmed") and not track.is_confirmed():
                    continue
            except Exception:
                pass

            track_id = str(getattr(track, "track_id", ""))
            if not track_id:
                continue

            det_label = None
            try:
                if hasattr(track, "get_det_class"):
                    det_label = track.get_det_class()
            except Exception:
                det_label = None
            if det_label is None:
                try:
                    det_label = getattr(track, "det_class", None)
                except Exception:
                    det_label = None
            if det_label is None:
                try:
                    det_label = getattr(track, "class_name", None)
                except Exception:
                    det_label = None
            try:
                if det_label is not None:
                    det_label = str(det_label)
            except Exception:
                det_label = None

            # prefer ltrb output
            ltrb = None
            try:
                if hasattr(track, "to_ltrb"):
                    ltrb = track.to_ltrb()
                elif hasattr(track, "to_tlbr"):
                    ltrb = track.to_tlbr()
            except Exception:
                ltrb = None

            if ltrb is None or len(ltrb) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in ltrb]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
            ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0

            score = 1.0
            try:
                if hasattr(track, "get_detection_confidence"):
                    conf = track.get_detection_confidence()
                    if conf is not None:
                        score = float(conf)
            except Exception:
                pass

            obj_data = {
                "label": det_label or "person",
                "score": float(score),
                "box": bbox,
                "area": int(area),
                "ratio": ratio,
                "region": bbox,
                "frame_time": frame_time,
            }

            # Frigate Stage A requires these fields for TrackedObject processing
            obj_data["estimate"] = bbox
            obj_data["estimate_velocity"] = (0, 0)
            obj_data["centroid"] = (
                int((bbox[0] + bbox[2]) / 2.0),
                int((bbox[1] + bbox[3]) / 2.0),
            )

            active_track_ids.add(track_id)

            if self.use_track_id_as_reid:
                # Track-based identity: never merges different people into one id.
                obj_data["reid_id"] = track_id
            else:
                # ReID-based identity: can merge tracks across time based on embeddings.
                new_features = None
                if bgr_frame is not None:
                    new_features = self.reid_extractor.extract_features(bgr_frame, bbox)
                    if new_features is not None:
                        self.track_features[track_id] = new_features

                identity_id = self._assign_identity(track_id, new_features)
                # Always expose a stable ID for downstream crop saving.
                # If ReID cannot be computed, fall back to the DeepSort track_id.
                if identity_id is None:
                    obj_data["reid_id"] = track_id
                else:
                    obj_data["reid_id"] = identity_id

            if track_id not in self.track_id_map:
                self.register(track_id, obj_data)
            else:
                self.update(track_id, obj_data)

            frigate_id = self.track_id_map.get(track_id)
            if frigate_id:
                # Keep reid_id in sync in tracked_objects store
                try:
                    self.tracked_objects[frigate_id]["reid_id"] = obj_data.get("reid_id")
                except Exception:
                    pass

        # Clean up disappeared tracks
        disappeared_tracks = set(self.track_id_map.keys()) - active_track_ids
        for track_id in disappeared_tracks:
            frigate_id = self.track_id_map[track_id]
            self.disappeared[frigate_id] += 1

            # Remove tracks that have been missing for too long
            if self.disappeared[frigate_id] > self.max_age:
                self.deregister(frigate_id, track_id)

        # Update list of boxes that are not yet tracked (used by Frigate consolidation)
        try:
            tracked_boxes = [obj.get("box") for obj in self.tracked_objects.values()]
            tracked_boxes = [b for b in tracked_boxes if isinstance(b, list) and len(b) == 4]
            untracked: list[list[int]] = []
            for d in detections:
                if isinstance(d, (tuple, list)) and len(d) > 2:
                    box = d[2]
                elif isinstance(d, dict):
                    box = d.get("box") or d.get("bbox") or d.get("region")
                else:
                    box = None

                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                box_list = [int(v) for v in box]
                if box_list in tracked_boxes:
                    continue
                untracked.append(box_list)
            self.untracked_object_boxes = untracked
        except Exception:
            self.untracked_object_boxes = []

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
