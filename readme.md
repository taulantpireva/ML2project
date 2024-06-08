Hello fellow student, have fun presenting

1. Project goal/Motivation

The goal of this project was to create a model that would help people track their food intake on a daily basis and help them on their fitness journey. The app should give people the opportunity to take pictures of their food and then get nutritional infor back on what they are eating. This would eliminate the task of scanning and weighing each and every meal they are eating. As a first step only the food and some basic information is displayed. The serving sizes and the nutritional values are not precise and somewhat arbitrary, but they are plausible.

    1.1 Further steps to make the app more user friendly
    As a further step the user should be prompted to take pictures of multiple angles so that a second model could then estimate the size of the plate and how much of each food is on it. This way we could give more precise information on the nutritional values of the meal.

2. Data Collection or Generation
   2.1 UECFOOD256
   For the data collection component of my project, I utilized the UECFOOD256 dataset, which is a comprehensive collection specifically designed for food recognition tasks. This dataset contains 256 distinct food categories and a total of 31,651 images, providing a diverse set of food items for training and evaluation. To prepare this dataset for use with the YOLOv5 framework, I created a script (convert_annotations_yolov.py) that converted the dataset into the YOLOv5 format, which includes generating annotation files in the required format and organizing the images into training, validation, and test directories. This preprocessing step was essential to ensure the dataset was properly structured for efficient training of the YOLOv5 model.

   2.2 FOOD101
   Additionally I wanted to use the Food 101 dataset from ETH Zurich, which is a comprehensive collection specifically designed for food recognition tasks, to further enhance the model. This dataset contains 101 distinct food categories and a total of 101,000 images, providing a diverse set of food items for training and evaluation. However, the Food-101 dataset did not include bounding box values, which are necessary for object detection tasks. To address this, I used a pre-trained ResNet50 model to generate bounding box values for the images. I created scripts (convert_annotations_food101.py/cam_annotation.py & change_annotation_id.py) that employed this model to predict the bounding boxes for each food item in the dataset. This preprocessing step was crucial to ensure the dataset was properly annotated and structured for efficient training of the YOLOv5 model. However due to time constraints I could not further fine tune my model with this dataset, as the training over 50 epochs would have taken days with my current hardware.

3. Modeling

For the modeling component of my project, I utilized the YOLOv5 object detection framework to fine-tune a pre-trained YOLOv5s model on my custom dataset. I began by cloning the YOLOv5 repository and installing all necessary dependencies. I then configured the dataset by creating a YAML file that mapped the custom categories and specified paths to the training, validation, and testing sets. The training process involved fine-tuning the YOLOv5s model using the dataset, adjusting the model weights over 50 epochs to enhance its performance in detecting and classifying food items. The training was conducted using a GPU when available to expedite the process. Upon completion, I saved the best-performing model weights for later use in the application, demonstrating a robust and methodical approach to model fine-tuning for object detection.

4. Project evaluation (The files of the training runs can be found in the folder "results")

Precision-Confidence Curve

The Precision-Confidence Curve shows the relationship between precision and the confidence levels of the model's predictions. Our model demonstrates an increasing precision with higher confidence levels, indicating that the predictions it makes with higher confidence are generally more accurate. The blue line illustrates a reasonable overall precision, though there is room for improvement to reduce variability at lower confidence levels.

Precision-Recall Curve

The Precision-Recall Curve highlights the trade-off between precision and recall. Our model achieves a mean Average Precision (mAP) of 0.368 at an Intersection over Union (IoU) threshold of 0.5. This indicates a balanced performance, though improvements could be made to enhance precision and recall simultaneously. The steep initial decline followed by a gradual decrease suggests that the model performs better with higher confidence predictions.

Recall-Confidence Curve

The Recall-Confidence Curve shows the relationship between recall and confidence levels. The model maintains a high recall at lower confidence thresholds, indicating it can identify a substantial proportion of actual positives. However, as confidence increases, recall decreases, which suggests that while the model is conservative at higher confidence levels, it might miss some positives. The overall shape of the curve suggests that the model strikes a balance between over- and under-detection.

Training Results

The Training Results provide a comprehensive overview of the model’s learning process. Key observations include:

Training Losses: Both the box, object, and classification losses steadily decrease over the epochs, indicating that the model is learning effectively.
Validation Losses: The validation losses also decrease, suggesting that the model generalizes well to unseen data.

Metrics: Precision, recall, and mAP metrics show significant improvement over the training period. The initial sharp rise followed by stabilization in precision and recall metrics indicates robust learning and convergence.

Overall Performance: The graphs collectively show that the model improves with training, achieving a balance between precision and recall while minimizing losses.

5. Additional information
   5.1 Comparison between the base model and the fine tuned one
   The base model performs reasonably well, but detects many things that are unnecessary for our application. The fine tuned model is more focused and performs better when detecting food specifically.

   5.2 Running the application
   Data sets are not provided but can be found here: https://www.kaggle.com/datasets/rkuo2000/uecfood256 and here:https://www.kaggle.com/datasets/kmader/food41.
   The application can be run locally or seen on streamlit: ml2project.streamlit.app.
   Test images can be found in the "ml2_testimages" folder or any image with food from the internet can be used.

   if you still wanna run the app locally:

   - clone the repo: https://github.com/taulantpireva/ML2project
   - open a terminal and go to the root directory "ml2project", then cd backend
   - run pip install -r requirements.txt
   - then run: streamlit run streamlit_app.py
   - open http://localhost:8501/ in your browser

     5.3 Hardware used
     To train the model I used a RTX 2070 Super which took around 4.5 hours for 50 epochs
