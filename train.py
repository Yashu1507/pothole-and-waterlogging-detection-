from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("Starting waterlogging training...")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    workers=2,
    project="runs",
    name="waterlogging_detector"
)

print("Training completed!")