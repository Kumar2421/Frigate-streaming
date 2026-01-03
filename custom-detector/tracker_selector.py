#!/usr/bin/env python3
"""
Tracker selector for Frigate
Allows switching between Norfair and DeepSORT trackers
"""

import os
import logging
from typing import Any

from frigate.config import DetectConfig
from frigate.track import ObjectTracker

logger = logging.getLogger(__name__)

def create_tracker(config: DetectConfig) -> ObjectTracker:
    """
    Create appropriate tracker based on configuration
    
    Args:
        config: Detection configuration
        
    Returns:
        Tracker instance
    """
    
    # Check if DeepSORT is enabled via environment variable
    use_deepsort = os.getenv('USE_DEEPSORT', 'false').lower() == 'true'
    
    if use_deepsort:
        try:
            from frigate.track.deepsort_tracker import DeepSORTFrigateTracker
            logger.info("Using DeepSORT tracker")
            # Create a dummy PTZ metrics object to avoid None errors
            from frigate.camera import PTZMetrics
            ptz_metrics = PTZMetrics(autotracker_enabled=False)
            return DeepSORTFrigateTracker(config, ptz_metrics)
        except ImportError as e:
            logger.warning(f"DeepSORT tracker not available: {e}, falling back to Norfair")
    
    # Default to Norfair tracker
    try:
        from frigate.track.norfair_tracker import NorfairTracker
        logger.info("Using Norfair tracker")
        # Create a dummy PTZ metrics object to avoid None errors
        from frigate.camera import PTZMetrics
        ptz_metrics = PTZMetrics(autotracker_enabled=False)
        return NorfairTracker(config, ptz_metrics)
    except ImportError as e:
        logger.error(f"Norfair tracker not available: {e}")
        raise RuntimeError("No tracking implementation available")
