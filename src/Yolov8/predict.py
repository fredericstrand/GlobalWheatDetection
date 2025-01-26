import torch
from ultralytics import YOLO
import os

# Load the trained model
model = YOLO('src/Yolov8/models/yolov8m/weights/best.pt')

os.makedirs("src/Yolov8/results", exist_ok=True)

# Single image prediction
results = model.predict(source='data/test/',
                        save=True,
                        save_dir="src/Yolov8/results/", 
                        conf=0.5)
