import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D
from ultralytics import YOLO
from src.Yolov8.data_parser import YOLODataset

base_model = tf.keras.models.load_model('yolov8m')

base_model.trainable = False

inputs = Input(shape=(640, 640, 3))

features = base_model(inputs, training=False)
x = Conv2D(265, (3,3), padding='same', activation='relu')(features)
x = Conv2D(128, (3,3), padding='same', activation='relu')(x)

bbox_output = Conv2D(4, (1,1), activation='sigmoid', name='bbox_output')(x)
conf_output = Conv2D(1, (1,1), activation='sigmoid', name='conf_output')(x)

model = Model(inputs, [bbox_output, conf_output])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = {
        'bbox_output': 'mse',
        'conf_output': 'binary_crossentropy'
    },
    metrics = {'confidence': 'accuracy'}
)

train_dataset = YOLODataset('data/train/images', 'data/train/labels')
val_dataset = YOLODataset('data/val/images', 'data/val/labels')

model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = 20
)
