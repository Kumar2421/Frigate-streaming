# YOLO 12 Setup Guide for Frigate

## Overview

This guide explains how to configure Frigate to use YOLO 12 for object detection and where the model performs in the project.

## Model Performance Locations

### 1. **Model Loading** (`frigate/detectors/plugins/onnx.py`)
   - **Location**: `ONNXDetector.__init__()`
   - **What happens**: The YOLO 12 ONNX model is loaded into ONNX Runtime
   - **Code**: Lines 33-65
   ```python
   self.model = ort.InferenceSession(path, providers=providers, provider_options=options)
   ```

### 2. **Model Inference** (`frigate/detectors/plugins/onnx.py`)
   - **Location**: `ONNXDetector.detect_raw()`
   - **What happens**: The model runs inference on preprocessed frames
   - **Code**: Lines 67-119
   ```python
   tensor_output = self.model.run(None, {model_input_name: tensor_input})
   return post_process_yolo(tensor_output, self.width, self.height)
   ```

### 3. **Input Preprocessing** (`frigate/object_detection/base.py`)
   - **Location**: `LocalObjectDetector._transform_input()`
   - **What happens**: Camera frames are resized, normalized, and formatted for YOLO
   - **Code**: Lines 62-72
   ```python
   tensor_input = np.transpose(tensor_input, self.input_transform)
   tensor_input = tensor_input.astype(np.float32)
   tensor_input /= 255  # Normalize to 0-1 range
   ```

### 4. **Detection Processing** (`frigate/object_detection/base.py`)
   - **Location**: `DetectorRunner.run()`
   - **What happens**: Frames from cameras are queued and sent to the model
   - **Code**: Lines 143-175
   ```python
   detections = object_detector.detect_raw(input_frame)
   ```

### 5. **Post-Processing** (`frigate/util/model.py`)
   - **Location**: `post_process_yolo()`
   - **What happens**: Raw model output is converted to bounding boxes and class IDs
   - **Code**: Lines 229-233
   ```python
   def post_process_yolo(output: list[np.ndarray], width: int, height: int):
       # Converts YOLO output format to Frigate detection format
   ```

### 6. **Result Distribution** (`frigate/object_detection/base.py`)
   - **Location**: `DetectorRunner.run()`
   - **What happens**: Detection results are written to shared memory and published
   - **Code**: Lines 171-172
   ```python
   self.outputs[connection_id]["np"][:] = detections[:]
   detector_publisher.publish(connection_id)
   ```

## Complete Detection Flow

```
Camera Feed
    ↓
FFmpeg Decoding
    ↓
Frame Extraction (frigate/camera/)
    ↓
Frame Preprocessing (frigate/object_detection/base.py)
    ↓
YOLO 12 Model Inference (frigate/detectors/plugins/onnx.py)
    ↓
Post-Processing (frigate/util/model.py)
    ↓
Detection Results
    ↓
Object Tracking (frigate/track/)
    ↓
Event Recording (frigate/events/)
```

## Downloading YOLO 12 ONNX Model

### Option 1: Export from Ultralytics (Recommended)

1. **Install Ultralytics**:
   ```bash
   pip install ultralytics
   ```

2. **Export YOLO 12 to ONNX**:
   ```python
   from ultralytics import YOLO
   
   # Load YOLO 12 model
   model = YOLO('yolo12n.pt')  # or yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt
   
   # Export to ONNX
   model.export(format='onnx', imgsz=640)  # Adjust imgsz if needed
   ```

3. **Move the exported model**:
   ```bash
   # The exported model will be in the current directory
   # Move it to your Frigate config directory
   move yolo12n.onnx config/model_cache/yolo12.onnx
   ```

### Option 2: Download Pre-exported Model

If YOLO 12 ONNX models are available online, download and place in:
```
config/model_cache/yolo12.onnx
```

## Configuration

Your `config/config.yml` is already configured for YOLO 12. Key settings:

```yaml
detectors:
  onnx:
    type: onnx
    device: AUTO  # Uses GPU if available, falls back to CPU
    model:
      model_type: yolo-generic
      width: 640
      height: 640
      input_tensor: nchw
      input_dtype: float
      input_pixel_format: bgr
      path: "config/model_cache/yolo12.onnx"
```

## Model Requirements

- **Format**: ONNX (.onnx)
- **Input Size**: 640x640 (default, can be adjusted)
- **Input Format**: NCHW (Batch, Channels, Height, Width)
- **Input Type**: Float32, normalized 0-1
- **Color Format**: BGR
- **Output Format**: Standard YOLO output (bounding boxes + class scores)

## Performance Notes

- **GPU Acceleration**: Set `device: GPU` if you have NVIDIA GPU with CUDA
- **CPU Performance**: YOLO 12 on CPU is faster than CPU detector but slower than GPU
- **Model Size**: Larger models (yolo12x) are more accurate but slower
- **Input Size**: Larger input (e.g., 1280x1280) improves accuracy but reduces speed

## Troubleshooting

1. **Model not found**: Ensure the ONNX file exists at the specified path
2. **Import error**: Install `onnxruntime`: `pip install onnxruntime`
3. **GPU not working**: Install `onnxruntime-gpu`: `pip install onnxruntime-gpu`
4. **Wrong output format**: Ensure model is exported with standard YOLO output format

## Model Performance Comparison

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|------|-------------|-------------|----------|
| YOLO12n | Smallest | Fastest | Fastest | Good |
| YOLO12s | Small | Fast | Fast | Better |
| YOLO12m | Medium | Medium | Medium | Best |
| YOLO12l | Large | Slow | Medium | Excellent |
| YOLO12x | Largest | Slowest | Slow | Excellent |

Choose based on your hardware and accuracy requirements.

