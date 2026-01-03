from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# Export with opset 20 (ONNX Runtime in Docker supports up to opset 21, but torch.onnx.export max is 20)
# This ensures compatibility with the Docker container's ONNX Runtime
model.export(format="onnx", imgsz=[640, 640], dynamic=True, opset=20, simplify=True)