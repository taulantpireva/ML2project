from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model
BASE_PATH = Path('D:/ZHAW/ML2/ML2project')
MODEL_PATH = BASE_PATH / 'backend/models/yolov5s_trained.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(MODEL_PATH), force_reload=True)

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

    # Draw bounding boxes on the image
    img_with_boxes = np.squeeze(results.render())
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    _, img_encoded = cv2.imencode('.jpg', img_with_boxes)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({'results': results_json, 'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
