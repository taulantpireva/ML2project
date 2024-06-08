import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm

# Load a pre-trained ResNet model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Generate Class Activation Map (CAM)
def generate_cam(image, model, target_class):
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get('layer4').register_forward_hook(hook_feature)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    output = model(image)
    _, predicted = torch.max(output, 1)

    final_conv_layer = features_blobs[0]
    num_ftrs = final_conv_layer.shape[1]

    cam = np.zeros((final_conv_layer.shape[2], final_conv_layer.shape[3]), dtype=np.float32)
    for i in range(num_ftrs):
        cam += weight_softmax[target_class][i] * final_conv_layer[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)  # Adding epsilon to avoid division by zero
    cam = cv2.resize(cam, (224, 224))

    return cam

# Convert CAM to bounding boxes
def cam_to_bboxes(cam, threshold=0.2):
    binary_cam = cam > threshold
    contours, _ = cv2.findContours(np.uint8(binary_cam), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    return bboxes

# Process images and update annotations
def process_images_and_update_annotations(data_path, start_index=0):
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(data_path, split)
        image_files = [f for f in os.listdir(split_path) if f.endswith('.jpg')]
        
        print(f"Processing {split} images...")
        for idx, image_file in enumerate(tqdm(image_files[start_index:], initial=start_index, total=len(image_files))):
            image_path = os.path.join(split_path, image_file)
            annotation_path = image_path.replace('.jpg', '.txt')
            
            # Skip if the annotation file already exists
            if os.path.exists(annotation_path):
                continue
            
            image = preprocess_image(image_path)
            
            target_class = 0  # Assume class 0 for all images; modify as necessary
            cam = generate_cam(image, model, target_class)
            bboxes = cam_to_bboxes(cam)
            
            # Update annotation file
            with open(annotation_path, 'w') as f:
                for bbox in bboxes:
                    x, y, w, h = bbox
                    img = cv2.imread(image_path)
                    img_height, img_width = img.shape[:2]
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")

    print("Annotations updated with CAM-generated bounding boxes.")

# Set the data path and process the images
data_path = 'food-101'
start_index = 0  # Set this to the index from where you want to continue processing
process_images_and_update_annotations(data_path, start_index=start_index)
