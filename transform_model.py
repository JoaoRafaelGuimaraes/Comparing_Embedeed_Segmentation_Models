from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("models/land-seg.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'