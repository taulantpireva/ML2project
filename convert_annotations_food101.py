import os
import random
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm  # For progress bar

# Base path (adjust as necessary)
BASE_PATH = Path('D:/ZHAW/ML2/ML2project')

# Paths
DATASET_PATH = BASE_PATH / 'food-101'
IMAGES_PATH = DATASET_PATH / 'images'
META_PATH = DATASET_PATH / 'meta'
TRAIN_PATH = DATASET_PATH / 'train'
VALID_PATH = DATASET_PATH / 'valid'
TEST_PATH = DATASET_PATH / 'test'

# Create directories if they don't exist
for path in [TRAIN_PATH, VALID_PATH, TEST_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Read category mapping
category_mapping = {}
labels_file = META_PATH / 'labels.txt'
with open(labels_file, 'r') as f:
    for idx, line in enumerate(f.readlines()):
        category_mapping[line.strip().lower().replace(' ', '_')] = idx  # Make sure to match the format

# Read image lists
train_images = []
with open(META_PATH / 'train.txt', 'r') as f:
    train_images = f.read().splitlines()

test_images = []
with open(META_PATH / 'test.txt', 'r') as f:
    test_images = f.read().splitlines()

# Function to create fake annotations
def create_fake_annotation(img_path, class_id):
    img = cv2.imread(str(img_path))
    img_height, img_width = img.shape[:2]

    # Fake bounding box covering the whole image
    x1, y1, x2, y2 = 0, 0, img_width, img_height
    x_center, y_center, width, height = convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height)

    return f"{class_id} {x_center} {y_center} {width} {height}\n"

# Function to convert bounding box to YOLO format
def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

# Process train images
print("Processing train images...")
for image_info in tqdm(train_images):
    try:
        class_name, img_name = image_info.split('/')
        class_name = class_name.lower().replace(' ', '_')
        class_id = category_mapping[class_name]
        img_path = IMAGES_PATH / class_name / f"{img_name}.jpg"

        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        # Create annotation file
        annotation = create_fake_annotation(img_path, class_id)
        annotation_path = TRAIN_PATH / f"{img_name}.txt"
        with open(annotation_path, 'w') as ann_file:
            ann_file.write(annotation)

        # Copy image to train directory
        shutil.copy(img_path, TRAIN_PATH / f"{img_name}.jpg")

    except KeyError as e:
        print(f"KeyError: {e} for class_name: {class_name}")

# Process test images
print("Processing test images...")
for image_info in tqdm(test_images):
    try:
        class_name, img_name = image_info.split('/')
        class_name = class_name.lower().replace(' ', '_')
        class_id = category_mapping[class_name]
        img_path = IMAGES_PATH / class_name / f"{img_name}.jpg"

        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        # Create annotation file
        annotation = create_fake_annotation(img_path, class_id)
        annotation_path = TEST_PATH / f"{img_name}.txt"
        with open(annotation_path, 'w') as ann_file:
            ann_file.write(annotation)

        # Copy image to test directory
        shutil.copy(img_path, TEST_PATH / f"{img_name}.jpg")

    except KeyError as e:
        print(f"KeyError: {e} for class_name: {class_name}")

# Split train data into train and validation sets
print("Splitting train data into train and validation sets...")
train_files = list(TRAIN_PATH.glob('*.jpg'))
random.shuffle(train_files)

valid_count = int(len(train_files) * 0.1)  # 10% for validation
valid_files = train_files[:valid_count]
train_files = train_files[valid_count:]

for file_path in valid_files:
    shutil.move(file_path, VALID_PATH / file_path.name)
    shutil.move(file_path.with_suffix('.txt'), VALID_PATH / file_path.with_suffix('.txt').name)

print("Annotations converted and data split into train, valid, and test sets.")
