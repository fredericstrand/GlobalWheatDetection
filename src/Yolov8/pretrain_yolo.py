from ultralytics import YOLO

base_model = YOLO('yolov8m.pt')

# Train the model
base_model.train(
    data='data/dataset.yaml',
    epochs=15,
    batch=16,
    imgsz=640,
    lr0=0.001,
    project='src/Yolov8/models/',
    name='yolov8m'
)

base_model.export(format='saved_model',
                  img_size=640,
                  dynamic=False,
                  project='src/Yolov8/models/',
                  name='yolov8m'
                  )