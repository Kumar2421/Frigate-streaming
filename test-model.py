import time
from ultralytics import YOLO

# Load the exported TensorRT engine
engine_path = "yolov8l.engine"
model = YOLO(engine_path)

# Dummy input (replace with a real image or tensor)
# For a real benchmark, use an actual image file on disk or a valid URL
input_source = "https://ultralytics.com/images/bus.jpg"

print(f"Running inference on: {engine_path}")
start = time.time()
results = model(input_source,save=True)
end = time.time()

print(f"Inference time: {end - start:.3f} s")
print("Results:", results)





# Inference time: 2.956 s yolol.pt
# Inference time: 1.481 s yolol.onnx
# Inference time: 0.731 s yolol.engine