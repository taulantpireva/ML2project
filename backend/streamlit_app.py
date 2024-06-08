import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from nutritional_info import nutritional_info  # Ensure this is the correct import

# Load models
BASE_PATH = Path('.').resolve()
MODEL_PATH = BASE_PATH / 'models' / 'yolov5s_trained.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH.resolve().as_posix(), force_reload=True)
base_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def clean_food_name(food_name):
    return food_name.split('\t')[-1].strip().lower()

# Streamlit app
st.title('Food Detection and Nutritional Information')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

serving_size = st.radio("Select Serving Size", ("small", "normal", "large"), index=1)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img = np.array(img)

    # Perform inference with fine-tuned model
    results = model(img)
    results_json = results.pandas().xyxy[0].to_json(orient="records")

    # Get detected food items and their nutritional information
    detected_foods = results.pandas().xyxy[0]['name'].tolist()
    cleaned_detected_foods = [clean_food_name(food) for food in detected_foods]
    st.markdown("### Fine Tuned Model Results")
    if not cleaned_detected_foods:
        st.write("No food detected for fine tuned model")
    else:
        nutritional_data = {food: nutritional_info.get(food, {}) for food in cleaned_detected_foods}

        # Draw bounding boxes on the image (fine-tuned model)
        img_with_boxes = np.squeeze(results.render())
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        st.image(img_with_boxes, caption='Processed Image', use_column_width=True)

        st.write("Detected foods (fine-tuned):")
        
        # Display nutritional information
        displayed_nutritional_info = {}
        for food, info in nutritional_data.items():
            factor = 1
            if serving_size == "small":
                factor = 0.8
            elif serving_size == "large":
                factor = 1.2
            displayed_nutritional_info[food] = {k: v * factor for k, v in info.items()}

        if displayed_nutritional_info:
            st.write("Nutritional information:")
            for food, info in displayed_nutritional_info.items():
                st.markdown(f"**{food.capitalize()}**")
                st.markdown(f"- **Protein**: {info.get('protein', 'N/A')}g")
                st.markdown(f"- **Carbs**: {info.get('carbs', 'N/A')}g")
                st.markdown(f"- **Fat**: {info.get('fat', 'N/A')}g")
                calories = info.get('protein', 0) * 4 + info.get('carbs', 0) * 4 + info.get('fat', 0) * 9
                st.markdown(f"- **Calories**: {calories}")
        else:
            st.write("No nutritional information available.")

    # Perform inference with base model
    base_results = base_model(img)
    base_img_with_boxes = np.squeeze(base_results.render())
    base_img_with_boxes = cv2.cvtColor(base_img_with_boxes, cv2.COLOR_BGR2RGB)

    st.markdown("### Base Model Results")
    st.image(base_img_with_boxes, caption='Base Model Processed Image', use_column_width=True)
