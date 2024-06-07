import torch
from pathlib import Path
import cv2
from matplotlib import pyplot as plt

# Paths
BASE_PATH = Path('D:/ZHAW/ML2/ML2project')
MODEL_PATH = BASE_PATH / 'backend/models/yolov5s_trained.pt'
IMG_PATH = BASE_PATH / 'path/to/test/image.jpg'  # Update with the path to your test image

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(MODEL_PATH), force_reload=True)

# Load image
img = cv2.imread(str(IMG_PATH))

# Perform inference
results = model(img)

# Display results
results.show()  # Show results in a window
results.save(Path('output'))  # Save results to 'output' directory

# Or plot with matplotlib
plt.imshow(cv2.cvtColor(results.render()[0], cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
