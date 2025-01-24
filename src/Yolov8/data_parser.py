import numpy as np
import os
from tensorflow.keras.utils import Sequence

class YOLODataset(Sequence):
    def __init__(self, image_dir, label_dir, input_size=(320, 320)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.image_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = tf.image.decode_image(tf.io.read_file(image_path))
        image = tf.image.resize(image, self.input_size) / 255.0

        # Load labels
        label_path = os.path.join(self.label_dir, self.image_files[index].replace('.jpg', '.txt'))
        with open(label_path, 'r') as file:
            labels = np.array([list(map(float, line.strip().split())) for line in file])

        bboxes = labels[:, 1:]  # x_center, y_center, width, height
        confidence = labels[:, 0:1]  # class_id

        return image, {'bbox_output': bboxes, 'conf_output': confidence}