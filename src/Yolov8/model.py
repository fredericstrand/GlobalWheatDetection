import tensorflow as tf
from tensorflow.keras import Model
from ultralytics import YOLO

base_model = YOLO('yolov8m.pt')

