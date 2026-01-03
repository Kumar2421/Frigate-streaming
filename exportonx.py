from ultralytics import YOLO
model = YOLO('yolo12n.pt')
model.export(format='onnx', imgsz=640, dynamic=True)