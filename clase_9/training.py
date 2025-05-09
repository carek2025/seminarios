from ultralytics import YOLO

model=YOLO('yolo11s.pt')

results=model.train(
    data='C:/Users/labin/Downloads/American Sign Language Letters.v1-v1.yolov11/data.yaml',
    epochs=500,
    batch=64,
    workers=8,
    imgsz=448,
    seed=4200,
    plots=True
)