from ultralytics import YOLO

model = YOLO("yolov8m.pt")


model.info()


results = model.train(data="fingers counting.v1i.yolov8\data.yaml", epochs=100, imgsz=640)