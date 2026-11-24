# PULMO AI – Pneumonia Detection Using CNN

## About the Project
PULMO AI is a web-based application developed using Python, Streamlit, and Convolutional Neural Networks (CNN). It analyzes chest X-ray images and predicts whether the image indicates NORMAL or PNEUMONIA.

This project is developed for academic and educational purposes.

## Features
- Upload chest X-ray images.
- Analyze images using a trained CNN model.
- Predict NORMAL or PNEUMONIA.
- Display prediction confidence.
- Simple and user-friendly interface.

## Technologies Used
- Python
- TensorFlow / Keras
- Convolutional Neural Network (CNN)
- Streamlit
- NumPy
- Pillow

## Project Files
- `app.py` – Main Streamlit application.
- `train_model.py` – Model training script.
- `evaluate_model.py` – Model evaluation script.
- `pneumonia_cnn_model.keras` – Trained CNN model.
- `class_names.json` – Class labels.
- `requirements.txt` – Required Python libraries.

## Dataset
Chest X-Ray Images (Pneumonia) Dataset:

https://data.mendeley.com/datasets/rscbjbr9sj/2

## How to Run the Project

1. Clone this repository.
2. Install the required dependencies:

   `pip install -r requirements.txt`

3. Run the application:

   `streamlit run app.py`

## Disclaimer
PULMO AI is an academic project and is not intended to replace professional medical advice, diagnosis, or treatment.
