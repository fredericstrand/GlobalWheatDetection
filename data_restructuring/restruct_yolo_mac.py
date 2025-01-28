import os
import pandas as pd
import shutil

# Define the path YOLO format data
csv_file = '../global-wheat-detection/train.csv'
images_dir = '../global-wheat-detection/train'
output_dir = '../data_yolo'

# Make the directories for the images and labels for the YOLO model
os.makedirs(f'{output_dir}/train/images', exist_ok=True)
os.makedirs(f'{output_dir}/val/images', exist_ok=True)
os.makedirs(f'{output_dir}/train/labels', exist_ok=True)
os.makedirs(f'{output_dir}/val/labels', exist_ok=True)

df = pd.read_csv(csv_file)

# Function to center coordinates, which is necessary for the YOLO model
def normalize_bbox(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width = width / img_width
    height = height / img_height
    return x_center, y_center, width, height

image_ids = df['image_id'].unique()
train_ids = image_ids[:int(len(image_ids)*0.8)]
val_ids = image_ids[int(len(image_ids)*0.8):]

print('Organizing images and labels for YOLO model. This may take a while...')
for _, row in df.iterrows():
    image_id = row['image_id']
    bbox = eval(row['bbox'])
    class_id = 0

    img_width, img_height = row['width'], row['height']
    x_center, y_center, width, height = normalize_bbox(bbox, img_width, img_height)

    split = 'train' if image_id in train_ids else 'val'

    label_path = f'{output_dir}/{split}/labels/{image_id}.txt'
    with open(label_path, 'a') as f:
        f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

    src_img_path = f'{images_dir}/{image_id}.jpg'
    dst_img_path = f'{output_dir}/{split}/images/{image_id}.jpg'

    if os.path.exists(src_img_path):
        shutil.copy(src_img_path, dst_img_path)

if os.path.exists('global-wheat-detection'):
    shutil.rmtree('global-wheat-detection')

os.makedirs('data/test', exist_ok=True)
