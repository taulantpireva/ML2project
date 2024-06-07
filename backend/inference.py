import subprocess
from pathlib import Path
import os

# Paths
BASE_PATH = Path('D:/ZHAW/ML2/ML2project')
YOLOV5_PATH = BASE_PATH / 'yolov5'  # Assuming yolov5 repo is cloned at the same level as backend
MODEL_PATH = BASE_PATH / 'backend/models/yolov5s_trained.pt'
IMG_PATH = BASE_PATH / 'backend/test_images/6100.jpg'  # Update with the path to your test image
OUTPUT_DIR = BASE_PATH / 'backend/test_images/output'  # Directory to save the annotated image

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Command to run detect.py
command = [
    str(BASE_PATH / '.venv/Scripts/python.exe'),  # Using the Python executable from the virtual environment
    str(YOLOV5_PATH / 'detect.py'),
    '--weights', str(MODEL_PATH),
    '--source', str(IMG_PATH),
    '--conf-thres', '0.15',  # Set the confidence threshold
    '--save-txt',
    '--save-conf',
    '--project', str(OUTPUT_DIR),
    '--name', 'results',  # Name of the results directory
    '--exist-ok'  # Allow existing directory
]

# Run the command
print("Running YOLOv5 detect.py...")
result = subprocess.run(command, capture_output=True, text=True)

# Check if the command was successful
if result.returncode == 0:
    print("YOLOv5 detection completed successfully.")
    print(result.stdout)
else:
    print("Error running YOLOv5 detection:")
    print(result.stderr)

# Display the results
from matplotlib import pyplot as plt
import cv2

# Path to the annotated image
annotated_img_path = OUTPUT_DIR / 'results' / IMG_PATH.name

# Load and display the image
img = cv2.imread(str(annotated_img_path))
if img is not None:
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print(f"Annotated image not found at {annotated_img_path}")
