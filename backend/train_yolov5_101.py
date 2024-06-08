import os
import shutil
import yaml
from pathlib import Path
import torch

# Base path (adjust as necessary)
BASE_PATH = Path('D:/ZHAW/ML2/ML2project')

# Paths
FOOD101_PATH = BASE_PATH / 'food-101'
BACKEND_PATH = BASE_PATH / 'backend'
MODELS_PATH = BACKEND_PATH / 'models'
YOLOV5_PATH = BASE_PATH / 'yolov5'

# Pre-trained model path
PRETRAINED_MODEL_PATH = MODELS_PATH / 'yolov5s_trained.pt'

# Create models directory if it doesn't exist
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Step 1: Prepare the YOLOv5 Environment
if not YOLOV5_PATH.exists():
    print("Cloning YOLOv5 repository...")
    os.system(f'git clone https://github.com/ultralytics/yolov5.git {YOLOV5_PATH}')

# Navigate to the YOLOV5 directory
os.chdir(YOLOV5_PATH)

# Install requirements
print("Installing YOLOv5 requirements...")
os.system('pip install -r requirements.txt')

# Step 2: Prepare the Dataset Configuration File
category_file = FOOD101_PATH / 'category.txt'
categories = {i: name for i, name in enumerate(open(category_file).read().splitlines())}

dataset_yaml = {
    'path': str(FOOD101_PATH),
    'train': str(FOOD101_PATH / 'train'),
    'val': str(FOOD101_PATH / 'valid'),
    'test': str(FOOD101_PATH / 'test'),
    'names': categories
}

with open('dataset_food101.yaml', 'w') as f:
    yaml.dump(dataset_yaml, f)

print("Dataset configuration file created:")
print(yaml.dump(dataset_yaml, default_flow_style=False))

# Step 3: Train/Fine-tune the Model
print("Starting the training process...")

# Check for GPU
device = '0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

os.system(f'python train.py --img 640 --batch 16 --epochs 5 --data dataset_food101.yaml --cfg yolov5s.yaml --weights yolov5s.pt --device {device} --name yolov5s_food101_results')

# Step 4: Save the Trained Model
# Find the latest training run directory
runs_dir = YOLOV5_PATH / 'runs/train'
latest_run_dir = max(runs_dir.glob('yolov5s_food101_results*'), key=os.path.getmtime)

weights_dir = latest_run_dir / 'weights'
trained_model_path = list(weights_dir.glob('best.pt'))
if trained_model_path:
    shutil.copy(trained_model_path[0], MODELS_PATH / 'yolov5_trained_food101.pt')
    print(f"Model saved to {MODELS_PATH / 'yolov5_trained_food101.pt'}")
else:
    print(f"Error: Trained model not found in {weights_dir}")

# Navigate back to the original directory
os.chdir(BASE_PATH)

print("Training completed and model saved.")
