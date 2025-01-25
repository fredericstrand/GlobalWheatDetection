from ultralytics import YOLO

base_model = YOLO('yolov8m.pt')

# Train the model
base_model.train(
    data='dataset.yaml',
    epochs=20,
    batch=8,
    imgsz=640,
    lr0=0.001,
    project='src/Yolov8/models/',
    name='yolov8m'
)

base_model.export(format='saved_model',
                  imgsz=640,
                  dynamic=False,
                  project='src/Yolov8/models/',
                  name='yolov8m'
                  )