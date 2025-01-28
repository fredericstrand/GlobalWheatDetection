import os
import pandas as pd
import shutil
import ast

# Define the paths for YOLO format data
csv_file = '../../global-wheat-detection/train.csv'
images_dir = '../../global-wheat-detection/train'
output_dir = 'data'

# Create directories for the YOLO model
os.makedirs(f'{output_dir}/train/images', exist_ok=True)
os.makedirs(f'{output_dir}/val/images', exist_ok=True)
os.makedirs(f'{output_dir}/train/labels', exist_ok=True)
os.makedirs(f'{output_dir}/val/labels', exist_ok=True)

# Load the dataset
df = pd.read_csv(csv_file)

# Function to normalize bounding boxes for YOLO format
def normalize_bbox(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width = width / img_width
    height = height / img_height
    return x_center, y_center, width, height

# Split dataset into train and validation sets
image_ids = df['image_id'].unique()
train_ids = set(image_ids[:int(len(image_ids) * 0.8)])
val_ids = set(image_ids[int(len(image_ids) * 0.8):])

print('Organizing images and labels for YOLO model. This may take a while...')
for _, row in df.iterrows():
    try:
        # Extract image and bounding box information
        image_id = row['image_id']
        bbox = ast.literal_eval(row['bbox'])  # Safely parse bbox
        class_id = 0  # Single class (wheat)
        img_width, img_height = row['width'], row['height']

        # Normalize bounding box
        x_center, y_center, width, height = normalize_bbox(bbox, img_width, img_height)

        # Determine split (train/val)
        split = 'train' if image_id in train_ids else 'val'

        # Write label file
        label_path = f'{output_dir}/{split}/labels/{image_id}.txt'
        with open(label_path, 'a') as f:
            f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

        # Copy image to corresponding folder
        src_img_path = os.path.join(images_dir, f'{image_id}.jpg')
        dst_img_path = os.path.join(f'{output_dir}/{split}/images', f'{image_id}.jpg')
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image {src_img_path} not found!")

    except Exception as e:
        print(f"Error processing row: {e}")

os.makedirs('data/test', exist_ok=True)
print("Dataset organized successfully.")
