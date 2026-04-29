import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohamedaelkhateb05/fingers-count-detection-yolov8")

print("Path to dataset files:", path)