from ultralytics import YOLO

base_model = YOLO('yolov8m.pt')

# Train the model
base_mode.train(
    data='data.yaml',
    epochs=20,
    batch_size=16,
    img_size=640,
    batch=8,
    lr=0.001,
    project='src/Yolov8/models/',
    name='yolov8m'
)

base_model.export(format='saved_model', img_size=640)