# Where the YOLO 12 Model Performs in Frigate

## 🎯 Quick Summary

The YOLO 12 model performs **object detection** - identifying and locating objects (person, car, etc.) in camera frames. Here's exactly where it runs in the codebase:

## 📍 Model Execution Points

### 1. **Model Loading** (Startup)
**File**: `frigate/detectors/plugins/onnx.py`  
**Function**: `ONNXDetector.__init__()` (Lines 33-65)  
**What it does**: 
- Loads the YOLO 12 ONNX model file into memory
- Initializes ONNX Runtime inference session
- Sets up GPU/CPU execution providers

```python
self.model = ort.InferenceSession(
    path, providers=providers, provider_options=options
)
```

### 2. **Frame Preprocessing** (Per Frame)
**File**: `frigate/object_detection/base.py`  
**Function**: `LocalObjectDetector._transform_input()` (Lines 62-72)  
**What it does**:
- Resizes camera frame to model input size (640x640)
- Converts color format (BGR)
- Normalizes pixel values (0-255 → 0-1)
- Transposes tensor format (NHWC → NCHW)

### 3. **Model Inference** (Per Frame) ⭐ **MAIN PERFORMANCE POINT**
**File**: `frigate/detectors/plugins/onnx.py`  
**Function**: `ONNXDetector.detect_raw()` (Lines 67-119)  
**What it does**:
- **Runs YOLO 12 model** on preprocessed frame
- Model analyzes the image and detects objects
- Returns raw detection results (bounding boxes, scores, class IDs)

```python
tensor_output = self.model.run(None, {model_input_name: tensor_input})
return post_process_yolo(tensor_output, self.width, self.height)
```

**This is where YOLO 12 actually performs detection!**

### 4. **Post-Processing** (Per Frame)
**File**: `frigate/util/model.py`  
**Function**: `post_process_yolo()` (Lines 229-233)  
**What it does**:
- Converts YOLO output format to Frigate format
- Applies Non-Maximum Suppression (NMS) to remove duplicate detections
- Filters detections by confidence threshold
- Normalizes bounding box coordinates

### 5. **Result Distribution** (Per Frame)
**File**: `frigate/object_detection/base.py`  
**Function**: `DetectorRunner.run()` (Lines 143-175)  
**What it does**:
- Writes detection results to shared memory
- Publishes results to other Frigate processes
- Used by tracking, recording, and event systems

## 🔄 Complete Detection Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Camera Feed (RTSP)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  FFmpeg Decoding (frigate/camera/)                          │
│  - Extracts frames from video stream                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Frame Preprocessing (frigate/object_detection/base.py)     │
│  - Resize to 640x640                                        │
│  - Normalize pixel values                                   │
│  - Format conversion (BGR, NCHW)                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  ⭐ YOLO 12 MODEL INFERENCE ⭐                              │
│  (frigate/detectors/plugins/onnx.py)                        │
│  - Model analyzes frame                                      │
│  - Detects objects (person, car, etc.)                      │
│  - Returns bounding boxes + confidence scores               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Post-Processing (frigate/util/model.py)                    │
│  - Apply NMS (remove duplicates)                           │
│  - Filter by confidence                                     │
│  - Normalize coordinates                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Result Distribution (frigate/object_detection/base.py)    │
│  - Write to shared memory                                   │
│  - Publish to other processes                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Object Tracking (frigate/track/)                           │
│  - Track objects across frames                              │
│  - Maintain object IDs                                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Event Recording (frigate/events/)                          │
│  - Create events for detected objects                       │
│  - Save snapshots and recordings                            │
└─────────────────────────────────────────────────────────────┘
```

## ⚡ Performance Characteristics

### Model Execution Frequency
- **Per Camera Frame**: Model runs once per frame sent for detection
- **Frame Rate**: Typically 5-10 FPS per camera (configurable)
- **Multiple Cameras**: Model processes frames from all cameras in queue

### Resource Usage
- **CPU**: Model inference uses CPU (or GPU if available)
- **Memory**: Model loaded once, reused for all frames
- **Speed**: 
  - CPU: ~50-200ms per frame (depends on model size)
  - GPU: ~10-50ms per frame (much faster)

### Optimization Tips
1. **Use GPU**: Set `device: GPU` in config for 5-10x speedup
2. **Model Size**: Smaller models (yolo12n) are faster but less accurate
3. **Input Size**: Smaller input (320x320) is faster but less accurate
4. **Multiple Detectors**: Run multiple detector processes for parallel processing

## 🔍 Key Code Locations

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Model Loading | `frigate/detectors/plugins/onnx.py` | 33-65 | Initialize ONNX model |
| **Model Inference** | `frigate/detectors/plugins/onnx.py` | 67-119 | **Run YOLO 12 detection** |
| Input Preprocessing | `frigate/object_detection/base.py` | 62-72 | Prepare frames for model |
| Detection Runner | `frigate/object_detection/base.py` | 143-175 | Process detection queue |
| Post-Processing | `frigate/util/model.py` | 229-233 | Convert YOLO output |
| YOLO Post-Process | `frigate/util/model.py` | 103-234 | NMS and filtering |

## 📊 Model Output Format

The YOLO 12 model outputs:
- **Bounding Boxes**: [x_min, y_min, x_max, y_max] (normalized 0-1)
- **Confidence Scores**: Probability that detection is correct (0-1)
- **Class IDs**: Object type (0=person, 2=car, etc.)

Frigate converts this to:
```python
[
    (class_name, confidence, (y_min, x_min, y_max, x_max)),
    ...
]
```

## 🎯 Summary

**The YOLO 12 model performs object detection in `frigate/detectors/plugins/onnx.py` at line 81:**

```python
tensor_output = self.model.run(None, {model_input_name: tensor_input})
```

This single line runs the entire YOLO 12 neural network on each camera frame, identifying all objects in the scene. The results are then processed, tracked, and recorded by other Frigate components.

