from ultralytics import YOLO
import multiprocessing

def train_model():
    model = YOLO("yolov8n.pt")
    results = model.train(
    data="Finger count.v2i.yolov8\data.yaml", 
    epochs=50,
    batch = 16,
    lr0 = 0.001, 
    imgsz=640,
    mosaic=1.0,    
    mixup=0.2,      
    copy_paste=0.2,
    degrees=20.0,   
    scale=0.5,       
    perspective=0.001, 
    dropout=0.1,     
    device=0   
)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_model()