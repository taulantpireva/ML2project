from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
from nutritional_info import nutritional_info

# Initialize Flask app
BASE_PATH = Path('D:/ZHAW/ML2/ML2project')
SVELTE_BUILD_PATH = BASE_PATH / 'svelte-app/public'
MODEL_PATH = BASE_PATH / 'backend/models/yolov5s_trained.pt'

app = Flask(__name__, static_folder=str(SVELTE_BUILD_PATH), static_url_path='')
CORS(app)  # Enable CORS for all routes

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(MODEL_PATH), force_reload=True)

def clean_food_name(food_name):
    return food_name.split('\t')[-1].strip().lower()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = np.array(img)

    # Perform inference
    results = model(img)
    results_json = results.pandas().xyxy[0].to_json(orient="records")

    # Get detected food items and their nutritional information
    detected_foods = results.pandas().xyxy[0]['name'].tolist()
    print(f"Detected foods: {detected_foods}")  # Print detected food names
    cleaned_detected_foods = [clean_food_name(food) for food in detected_foods]
    print(f"Cleaned foods: {cleaned_detected_foods}")  # Print cleaned food names
    nutritional_data = {food: nutritional_info.get(food, {}) for food in cleaned_detected_foods}
    print(nutritional_data)

    # Draw bounding boxes on the image
    img_with_boxes = np.squeeze(results.render())
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    _, img_encoded = cv2.imencode('.jpg', img_with_boxes)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({'results': results_json, 'image': img_base64, 'nutritional_info': nutritional_data})

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
