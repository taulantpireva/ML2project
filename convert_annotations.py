import os
import random
from pathlib import Path
import shutil
import cv2

# Base path (adjust as necessary)
BASE_PATH = Path('D:/ZHAW/ML2/ML2project')

# Paths
DATASET_PATH = BASE_PATH / 'dataset256/UECFOOD256'
LABELS_PATH = DATASET_PATH / 'labels'
TRAIN_PATH = DATASET_PATH / 'train'
VALID_PATH = DATASET_PATH / 'valid'
TEST_PATH = DATASET_PATH / 'test'

# Create directories if they don't exist
for path in [LABELS_PATH, TRAIN_PATH, VALID_PATH, TEST_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Read category mapping
category_mapping = {}
category_file = DATASET_PATH / 'category.txt'
with open(category_file, 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            class_id, class_name = parts
            category_mapping[class_name] = int(class_id) - 1  # 0-indexed

# Function to convert bounding box to YOLO format
def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

# Process each category folder
for category_id in range(1, 257):
    category_path = DATASET_PATH / str(category_id)
    bb_info_file = category_path / 'bb_info.txt'
    
    with open(bb_info_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip the header line
    
    for line in lines:
        parts = line.strip().split()
        img_name = f"{parts[0]}.jpg"
        x1, y1, x2, y2 = map(int, parts[1:])
        
        img_path = category_path / img_name
        if not img_path.exists():
            continue
        
        # Read actual image dimensions
        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]
        
        # Convert to YOLO format
        x_center, y_center, width, height = convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height)
        
        # Check for valid coordinates
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            print(f"Skipping invalid annotation in {img_name}")
            continue
        
        # Create annotation file
        annotation_path = LABELS_PATH / img_name.replace('.jpg', '.txt')
        with open(annotation_path, 'a') as ann_file:
            ann_file.write(f"{category_id - 1} {x_center} {y_center} {width} {height}\n")
        
        # Split into train, valid, and test
        dest_dir = random.choices([TRAIN_PATH, VALID_PATH, TEST_PATH], weights=[80, 10, 10], k=1)[0]
        shutil.copy(img_path, dest_dir / img_name)
        shutil.copy(annotation_path, dest_dir / annotation_path.name)

print("Annotations converted and data split into train, valid, and test sets.")
