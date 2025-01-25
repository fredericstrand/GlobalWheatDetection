import os
import torch
import numpy as np
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from PIL import Image

class YOLODataset(Dataset):
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
        image = Image.open(image_path).convert("RGB")
        image = F.resize(image, self.input_size)  # Resize to input_size
        image = F.to_tensor(image)  # Convert to tensor and scale to [0, 1]

        # Load labels
        label_path = os.path.join(self.label_dir, self.image_files[index].replace('.jpg', '.txt'))
        if not os.path.exists(label_path) or os.stat(label_path).st_size == 0:
            # Return empty bounding boxes and confidence if label file is missing or empty
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            confidence = torch.zeros((0, 1), dtype=torch.float32)
        else:
            with open(label_path, 'r') as file:
                labels = np.array([list(map(float, line.strip().split())) for line in file])
            bboxes = torch.tensor(labels[:, 1:], dtype=torch.float32)  # x_center, y_center, width, height
            confidence = torch.tensor(labels[:, 0:1], dtype=torch.float32)  # class_id

        return image, {'bbox': bboxes, 'confidence': confidence}


# Custom collate function
def custom_collate_fn(batch):
    images = []
    bboxes = []
    confidences = []

    for image, target in batch:
        images.append(image)
        bboxes.append(target['bbox'])
        confidences.append(target['confidence'])

    images = torch.stack(images, dim=0)
    return images, {'bbox': bboxes, 'confidence': confidences}