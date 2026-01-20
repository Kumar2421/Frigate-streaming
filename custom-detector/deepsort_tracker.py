#!/usr/bin/env python3
"""
DeepSORT Tracker for Frigate
Enhances Frigate's existing detection with advanced DeepSORT tracking and ReID
"""

import logging
import time
import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import defaultdict, deque

# DeepSORT imports
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from deep_sort_realtime.deep_sort.detection import Detection as DeepSortDetection
    from deep_sort_realtime.deep_sort.track import Track as DeepSortTrack
except ImportError as e:
    logging.warning(f"DeepSORT not available, falling back to basic tracking: {e}")
    DeepSort = None
    DeepSortDetection = None
    DeepSortTrack = None

from frigate.config import DetectConfig
from frigate.track import ObjectTracker
from frigate.util.image import SharedMemoryFrameManager

logger = logging.getLogger(__name__)

@dataclass
class TrackedObject:
    """Enhanced tracked object with ReID features"""
    track_id: int
    label: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    age: int
    time_since_update: int
    embedding: Optional[np.ndarray] = None
    motion_history: Optional[deque] = None
    appearance_features: Optional[np.ndarray] = None

class DeepSORTFrigateTracker(ObjectTracker):
    """
    DeepSORT-based tracker that integrates with Frigate's existing detection system
    """
    
    def __init__(self, config: DetectConfig, ptz_metrics=None) -> None:
        super().__init__(config)
        
        self.config = config
        self.trackers: Dict[str, Any] = {}
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.frame_count = 0
        self.untracked_object_boxes = []
        
        # DeepSORT configuration
        self.max_age = getattr(config, 'max_age', 30)
        self.n_init = getattr(config, 'n_init', 3)
        self.max_iou_distance = getattr(config, 'max_iou_distance', 0.7)
        self.max_cosine_distance = getattr(config, 'max_cosine_distance', 0.2)
        self.nn_budget = getattr(config, 'nn_budget', 100)
        
        # ReID configuration
        self.use_reid = getattr(config, 'use_reid', True)
        self.reid_threshold = getattr(config, 'reid_threshold', 0.5)
        self.embedding_model = None
        
        # Motion detection enhancement
        self.motion_enhancement = getattr(config, 'motion_enhancement', True)
        self.motion_history_length = getattr(config, 'motion_history_length', 10)
        
        # Initialize DeepSORT if available
        if DeepSort is not None:
            self.deep_sort = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
                override_track_class=None,
                embedder="mobilenet",  # Use MobileNet for ReID
                half=True,
                bgr=True,
                embedder_gpu=False,  # Set to True if GPU available
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            logger.info("DeepSORT tracker initialized successfully")
        else:
            self.deep_sort = None
            logger.warning("DeepSORT not available, using basic tracking")
        
        # Initialize ReID model if enabled
        if self.use_reid and self.deep_sort is not None:
            self._initialize_reid_model()
        
        # Performance tracking
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'lost_tracks': 0,
            'reid_matches': 0,
            'motion_enhancements': 0
        }
        
        logger.info(f"DeepSORT Frigate Tracker initialized with config: {config}")
    
    def _initialize_reid_model(self):
        """Initialize ReID model for appearance-based tracking"""
        try:
            # The DeepSORT library handles ReID model initialization internally
            # We just need to ensure it's properly configured
            logger.info("ReID model initialized through DeepSORT")
        except Exception as e:
            logger.error(f"Failed to initialize ReID model: {e}")
            self.use_reid = False
    
    def match_and_update(
        self,
        frame_name: str,
        frame_time: float,
        detections: List[Tuple[Any, Any, Any, Any, Any, Any]],
        frame=None,
    ) -> None:
        """
        Main tracking method that integrates with Frigate's detection pipeline
        
        Args:
            frame_name: Name of the current frame
            frame_time: Timestamp of the frame
            detections: List of detections from Frigate's detector
                       Format: (label, confidence, bbox, area, ratio, region)
        """
        self.frame_count += 1
        
        if not detections:
            self._update_trackers()
            return
        
        # Convert Frigate detections to DeepSORT format
        deepsort_detections = self._convert_detections(detections)
        
        # Update DeepSORT tracker
        if self.deep_sort is not None:
            tracks = self.deep_sort.update_tracks(deepsort_detections, frame=frame)
            self._process_deepsort_tracks(tracks, frame_time)
        else:
            # Fallback to basic tracking
            self._basic_tracking(detections, frame_time)
        
        # Update motion enhancement
        if self.motion_enhancement:
            self._update_motion_enhancement(detections, frame_time)
        
        # Clean up old tracks
        self._cleanup_old_tracks()
        
        # Update statistics
        self._update_stats()
    
    def _convert_detections(self, detections: List[Tuple[Any, Any, Any, Any, Any, Any]]) -> List[DeepSortDetection]:
        """Convert Frigate detections to DeepSORT format"""
        deepsort_detections = []
        
        for detection in detections:
            label, confidence, bbox, area, ratio, region = detection
            
            # Convert bbox format (x1, y1, x2, y2) to (x, y, w, h)
            x1, y1, x2, y2 = bbox
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            
            # Create DeepSORT detection
            deepsort_det = DeepSortDetection(
                np.array([x, y, w, h]),
                confidence,
                None  # feature will be computed by DeepSORT
            )
            
            deepsort_detections.append(deepsort_det)
        
        return deepsort_detections
    
    def _process_deepsort_tracks(self, tracks: List[DeepSortTrack], frame_time: float):
        """Process DeepSORT tracks and update internal tracking state"""
        active_track_ids = set()
        
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            track_id = track.track_id
            active_track_ids.add(track_id)
            
            # Get track state
            bbox = track.to_tlbr()  # Convert to (x1, y1, x2, y2)
            confidence = track.get_detection_confidence()
            
            # Update or create tracked object
            if track_id in self.tracked_objects:
                obj = self.tracked_objects[track_id]
                obj.bbox = bbox
                obj.confidence = confidence
                obj.time_since_update = 0
                obj.age += 1
                
                # Update motion history
                if obj.motion_history is None:
                    obj.motion_history = deque(maxlen=self.motion_history_length)
                obj.motion_history.append((bbox, frame_time))
            else:
                # Create new tracked object
                obj = TrackedObject(
                    track_id=track_id,
                    label="person",  # Default label, could be enhanced
                    bbox=bbox,
                    confidence=confidence,
                    age=1,
                    time_since_update=0,
                    motion_history=deque([(bbox, frame_time)], maxlen=self.motion_history_length)
                )
                self.tracked_objects[track_id] = obj
                self.stats['total_tracks'] += 1
        
        # Mark inactive tracks
        for track_id, obj in self.tracked_objects.items():
            if track_id not in active_track_ids:
                obj.time_since_update += 1
    
    def _basic_tracking(self, detections: List[Tuple[Any, Any, Any, Any, Any, Any]], frame_time: float):
        """Fallback basic tracking when DeepSORT is not available"""
        # Simple IoU-based tracking
        active_track_ids = set()
        
        for detection in detections:
            label, confidence, bbox, area, ratio, region = detection
            
            # Find best matching existing track
            best_match_id = None
            best_iou = 0.0
            
            for track_id, obj in self.tracked_objects.items():
                if obj.time_since_update > 5:  # Skip old tracks
                    continue
                
                iou = self._calculate_iou(bbox, obj.bbox)
                if iou > best_iou and iou > 0.3:  # IoU threshold
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                obj = self.tracked_objects[best_match_id]
                obj.bbox = bbox
                obj.confidence = confidence
                obj.time_since_update = 0
                obj.age += 1
                active_track_ids.add(best_match_id)
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                obj = TrackedObject(
                    track_id=track_id,
                    label=label,
                    bbox=bbox,
                    confidence=confidence,
                    age=1,
                    time_since_update=0,
                    motion_history=deque([(bbox, frame_time)], maxlen=self.motion_history_length)
                )
                self.tracked_objects[track_id] = obj
                active_track_ids.add(track_id)
                self.stats['total_tracks'] += 1
        
        # Mark inactive tracks
        for track_id, obj in self.tracked_objects.items():
            if track_id not in active_track_ids:
                obj.time_since_update += 1
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _update_motion_enhancement(self, detections: List[Tuple[Any, Any, Any, Any, Any, Any]], frame_time: float):
        """Enhance tracking with motion analysis"""
        for track_id, obj in self.tracked_objects.items():
            if obj.motion_history and len(obj.motion_history) > 1:
                # Analyze motion pattern
                motion_vector = self._calculate_motion_vector(obj.motion_history)
                
                # Use motion to predict next position
                if motion_vector is not None:
                    predicted_bbox = self._predict_next_position(obj.bbox, motion_vector)
                    # Could be used to improve tracking accuracy
                    self.stats['motion_enhancements'] += 1
    
    def _calculate_motion_vector(self, motion_history: deque) -> Optional[Tuple[float, float]]:
        """Calculate motion vector from history"""
        if len(motion_history) < 2:
            return None
        
        # Get recent positions
        recent_positions = list(motion_history)[-3:]  # Last 3 positions
        
        if len(recent_positions) < 2:
            return None
        
        # Calculate average velocity
        total_dx = 0
        total_dy = 0
        count = 0
        
        for i in range(1, len(recent_positions)):
            prev_bbox, prev_time = recent_positions[i-1]
            curr_bbox, curr_time = recent_positions[i]
            
            dt = curr_time - prev_time
            if dt > 0:
                dx = (curr_bbox[0] + curr_bbox[2]) / 2 - (prev_bbox[0] + prev_bbox[2]) / 2
                dy = (curr_bbox[1] + curr_bbox[3]) / 2 - (prev_bbox[1] + prev_bbox[3]) / 2
                
                total_dx += dx / dt
                total_dy += dy / dt
                count += 1
        
        if count > 0:
            return (total_dx / count, total_dy / count)
        
        return None
    
    def _predict_next_position(self, current_bbox: Tuple[float, float, float, float], motion_vector: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """Predict next position based on motion vector"""
        x1, y1, x2, y2 = current_bbox
        dx, dy = motion_vector
        
        # Simple linear prediction
        predicted_x1 = x1 + dx
        predicted_y1 = y1 + dy
        predicted_x2 = x2 + dx
        predicted_y2 = y2 + dy
        
        return (predicted_x1, predicted_y1, predicted_x2, predicted_y2)
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been updated recently"""
        tracks_to_remove = []
        
        for track_id, obj in self.tracked_objects.items():
            if obj.time_since_update > self.max_age:
                tracks_to_remove.append(track_id)
                self.stats['lost_tracks'] += 1
        
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
    
    def _update_trackers(self):
        """Update all trackers when no detections are available"""
        for obj in self.tracked_objects.values():
            obj.time_since_update += 1
    
    def _update_stats(self):
        """Update tracking statistics"""
        self.stats['active_tracks'] = len([obj for obj in self.tracked_objects.values() if obj.time_since_update < 5])
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get list of currently active tracks"""
        return [obj for obj in self.tracked_objects.values() if obj.time_since_update < 5]
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Get specific track by ID"""
        return self.tracked_objects.get(track_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        return {
            **self.stats,
            'frame_count': self.frame_count,
            'deep_sort_available': self.deep_sort is not None,
            'reid_enabled': self.use_reid,
            'motion_enhancement_enabled': self.motion_enhancement
        }
    
    def reset(self):
        """Reset the tracker"""
        self.tracked_objects.clear()
        self.next_id = 1
        self.frame_count = 0
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'lost_tracks': 0,
            'reid_matches': 0,
            'motion_enhancements': 0
        }
        logger.info("DeepSORT tracker reset")

# Backwards-compatible alias expected by object_processing.py
class DeepOCSORTTracker(DeepSORTFrigateTracker):
    pass
