import os
import shutil
import yaml
from pathlib import Path
import torch

# Base path (adjust as necessary)
BASE_PATH = Path('D:/ZHAW/ML2/ML2project')

# Paths
DATASET_PATH = BASE_PATH / 'dataset256/UECFOOD256'
BACKEND_PATH = BASE_PATH / 'backend'
MODELS_PATH = BACKEND_PATH / 'models'
YOLOV5_PATH = BASE_PATH / 'yolov5'

# Create models directory if it doesn't exist
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Step 1: Prepare the YOLOv5 Environment
if not YOLOV5_PATH.exists():
    print("Cloning YOLOv5 repository...")
    os.system(f'git clone https://github.com/ultralytics/yolov5.git {YOLOV5_PATH}')

# Navigate to the YOLOv5 directory
os.chdir(YOLOV5_PATH)

# Install requirements
print("Installing YOLOv5 requirements...")
os.system('pip install -r requirements.txt')

# Step 2: Prepare the Dataset Configuration File
category_file = DATASET_PATH / 'category.txt'
categories = {i-1: name for i, name in enumerate(open(category_file).read().splitlines()[1:], start=1)}

dataset_yaml = {
    'path': str(DATASET_PATH),
    'train': str(DATASET_PATH / 'train'),
    'val': str(DATASET_PATH / 'valid'),
    'test': str(DATASET_PATH / 'test'),
    'names': categories
}

with open('dataset.yaml', 'w') as f:
    yaml.dump(dataset_yaml, f)

# Step 3: Train/Fine-tune the Model
print("Starting the training process...")

# Check for GPU
device = '0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

os.system(f'python train.py --img 640 --batch 16 --epochs 1 --data dataset.yaml --cfg yolov5s.yaml --weights yolov5s.pt --device {device} --name yolov5s_results')

# Step 4: Save the Trained Model
# Find the latest training run directory
runs_dir = YOLOV5_PATH / 'runs/train'
latest_run_dir = max(runs_dir.glob('yolov5s_results*'), key=os.path.getmtime)

weights_dir = latest_run_dir / 'weights'
trained_model_path = list(weights_dir.glob('best.pt'))
if trained_model_path:
    shutil.copy(trained_model_path[0], MODELS_PATH / 'yolov5s_trained.pt')
    print(f"Model saved to {MODELS_PATH / 'yolov5s_trained.pt'}")
else:
    print(f"Error: Trained model not found in {weights_dir}")

# Navigate back to the original directory
os.chdir(BASE_PATH)

print("Training completed and model saved.")
