#!/usr/bin/env python3
"""
Patch script to integrate DeepSORT tracker with Frigate
This script modifies Frigate's tracking system to use our custom DeepSORT tracker
"""

import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_frigate():
    """Apply patches to integrate DeepSORT tracker"""
    
    # Paths
    frigate_track_dir = "/opt/frigate/frigate/track"
    custom_tracker_path = "/opt/frigate/custom_trackers/deepsort_tracker.py"
    backup_dir = "/opt/frigate/backup"
    
    try:
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup original files
        logger.info("Creating backups of original Frigate files...")
        shutil.copy2(f"{frigate_track_dir}/norfair_tracker.py", f"{backup_dir}/norfair_tracker.py.backup")
        shutil.copy2(f"{frigate_track_dir}/__init__.py", f"{backup_dir}/__init__.py.backup")
        
        # Copy our custom tracker
        logger.info("Installing custom DeepSORT tracker...")
        shutil.copy2(custom_tracker_path, f"{frigate_track_dir}/deepsort_tracker.py")
        
        # Modify __init__.py to include our tracker
        logger.info("Modifying Frigate's track module...")
        modify_track_init(frigate_track_dir)
        
        # Create a modified version of video.py that uses our tracker
        logger.info("Creating tracker selection mechanism...")
        create_tracker_selector(frigate_track_dir)
        
        logger.info("DeepSORT integration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error patching Frigate: {e}")
        # Restore backups on error
        restore_backups(backup_dir, frigate_track_dir)
        raise

def modify_track_init(track_dir):
    """Modify the track module's __init__.py to include our tracker"""
    
    init_file = f"{track_dir}/__init__.py"
    
    # Read current content
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Add import for our custom tracker
    new_content = content + """
# Custom DeepSORT tracker
from .deepsort_tracker import DeepSORTFrigateTracker

# Export the custom tracker
__all__ = ['ObjectTracker', 'DeepSORTFrigateTracker']
"""
    
    # Write modified content
    with open(init_file, 'w') as f:
        f.write(new_content)

def create_tracker_selector(track_dir):
    """Create a tracker selector that can choose between Norfair and DeepSORT"""
    
    selector_content = '''#!/usr/bin/env python3
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
    
    # Check if DeepSORT is enabled in config
    use_deepsort = getattr(config, 'use_deepsort', False)
    
    if use_deepsort:
        try:
            from frigate.track.deepsort_tracker import DeepSORTFrigateTracker
            logger.info("Using DeepSORT tracker")
            return DeepSORTFrigateTracker(config)
        except ImportError as e:
            logger.warning(f"DeepSORT tracker not available: {e}, falling back to Norfair")
    
    # Default to Norfair tracker
    try:
        from frigate.track.norfair_tracker import NorfairTracker
        logger.info("Using Norfair tracker")
        return NorfairTracker(config, None)  # PTZ metrics not needed for basic tracking
    except ImportError as e:
        logger.error(f"Norfair tracker not available: {e}")
        raise RuntimeError("No tracking implementation available")
'''
    
    # Write selector file
    with open(f"{track_dir}/tracker_selector.py", 'w') as f:
        f.write(selector_content)

def restore_backups(backup_dir, track_dir):
    """Restore original files from backup"""
    logger.info("Restoring original Frigate files...")
    
    try:
        shutil.copy2(f"{backup_dir}/norfair_tracker.py.backup", f"{track_dir}/norfair_tracker.py")
        shutil.copy2(f"{backup_dir}/__init__.py.backup", f"{track_dir}/__init__.py")
        logger.info("Backup restoration completed")
    except Exception as e:
        logger.error(f"Error restoring backups: {e}")

if __name__ == "__main__":
    patch_frigate()
