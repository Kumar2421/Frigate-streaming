#!/usr/bin/env python3
"""
Integration script to merge DeepSORT configuration with Frigate's main config
"""

import yaml
import os
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return {}

def save_yaml_config(config: Dict[str, Any], file_path: str):
    """Save configuration to YAML file"""
    try:
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving {file_path}: {e}")

def integrate_deepsort_config():
    """Integrate DeepSORT configuration with Frigate's main config"""
    
    # Paths
    main_config_path = "/config/config.yml"
    deepsort_config_path = "/config/deepsort_config.yml"
    backup_path = "/config/config.yml.backup"
    
    # Load configurations
    logger.info("Loading Frigate main configuration...")
    main_config = load_yaml_config(main_config_path)
    
    logger.info("Loading DeepSORT configuration...")
    deepsort_config = load_yaml_config(deepsort_config_path)
    
    if not main_config:
        logger.error("Failed to load main configuration")
        return False
    
    if not deepsort_config:
        logger.warning("DeepSORT configuration not found, using defaults")
        deepsort_config = get_default_deepsort_config()
    
    # Create backup
    logger.info("Creating backup of main configuration...")
    save_yaml_config(main_config, backup_path)
    
    # Integrate DeepSORT settings
    logger.info("Integrating DeepSORT configuration...")
    
    # Add global DeepSORT settings
    if 'deepsort' in deepsort_config:
        main_config['deepsort'] = deepsort_config['deepsort']
    
    # Add camera-specific tracking settings
    if 'cameras' in deepsort_config and 'cameras' in main_config:
        for camera_name, camera_deepsort_config in deepsort_config['cameras'].items():
            if camera_name in main_config['cameras']:
                # Add tracking configuration to camera
                if 'tracking' in camera_deepsort_config:
                    main_config['cameras'][camera_name]['tracking'] = camera_deepsort_config['tracking']
                
                # Enable DeepSORT for this camera
                main_config['cameras'][camera_name]['use_deepsort'] = True
    
    # Add advanced tracking features
    if 'advanced_tracking' in deepsort_config:
        main_config['advanced_tracking'] = deepsort_config['advanced_tracking']
    
    # Add detector configuration for DeepSORT
    if 'detectors' not in main_config:
        main_config['detectors'] = {}
    
    # Add DeepSORT detector configuration
    main_config['detectors']['deepsort'] = {
        'type': 'deepsort',
        'model': {
            'path': '/opt/frigate/custom_trackers/deepsort_tracker.py',
            'width': 320,
            'height': 320
        },
        'max_age': deepsort_config.get('deepsort', {}).get('max_age', 30),
        'n_init': deepsort_config.get('deepsort', {}).get('n_init', 3),
        'max_iou_distance': deepsort_config.get('deepsort', {}).get('max_iou_distance', 0.7),
        'max_cosine_distance': deepsort_config.get('deepsort', {}).get('max_cosine_distance', 0.2),
        'nn_budget': deepsort_config.get('deepsort', {}).get('nn_budget', 100),
        'use_reid': deepsort_config.get('deepsort', {}).get('use_reid', True),
        'motion_enhancement': deepsort_config.get('deepsort', {}).get('motion_enhancement', True)
    }
    
    # Save integrated configuration
    logger.info("Saving integrated configuration...")
    save_yaml_config(main_config, main_config_path)
    
    logger.info("DeepSORT integration completed successfully!")
    return True

def get_default_deepsort_config() -> Dict[str, Any]:
    """Get default DeepSORT configuration"""
    return {
        'deepsort': {
            'enabled': True,
            'max_age': 30,
            'n_init': 3,
            'max_iou_distance': 0.7,
            'max_cosine_distance': 0.2,
            'nn_budget': 100,
            'use_reid': True,
            'reid_threshold': 0.5,
            'embedder': 'mobilenet',
            'motion_enhancement': True,
            'motion_history_length': 10,
            'half_precision': True,
            'gpu_enabled': False
        },
        'cameras': {
            'em-dept-exit': {
                'tracking': {
                    'use_deepsort': True,
                    'max_age': 30,
                    'n_init': 3,
                    'motion_enhancement': True
                }
            },
            'em-dept-entry': {
                'tracking': {
                    'use_deepsort': True,
                    'max_age': 25,
                    'n_init': 2,
                    'motion_enhancement': True
                }
            },
            'em-dept-rightexit': {
                'tracking': {
                    'use_deepsort': True,
                    'max_age': 35,
                    'n_init': 4,
                    'motion_enhancement': True
                }
            },
            'em-dept-sideexit': {
                'tracking': {
                    'use_deepsort': True,
                    'max_age': 30,
                    'n_init': 3,
                    'motion_enhancement': True
                }
            }
        },
        'advanced_tracking': {
            'multi_camera': False,
            'cross_camera_tracking': False,
            'reid_features': {
                'enabled': True,
                'model_size': 'small',
                'update_interval': 5
            },
            'motion_prediction': {
                'enabled': True,
                'prediction_frames': 3,
                'kalman_filter': True
            }
        }
    }

if __name__ == "__main__":
    success = integrate_deepsort_config()
    if success:
        print("DeepSORT integration completed successfully!")
    else:
        print("DeepSORT integration failed!")
        exit(1)
