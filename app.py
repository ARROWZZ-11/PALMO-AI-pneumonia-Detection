import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Pneumonia Detection | CNN",
    page_icon="🫁",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #f5f7fa;
    color: #263445;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* General text color */

.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown label,
.stMarkdown span {
    color: #263445;
}

/* Headings */

h1, h2, h3, h4 {
    color: #24527a !important;
}

/* Main title */

.main-title {
    text-align: center;
    color: #24527a;
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #52677e !important;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #eaf1f8;
    border-right: 1px solid #dce5ef;
}

section[data-testid="stSidebar"] * {
    color: #263445;
}

section[data-testid="stSidebar"] h1 {
    color: #24527a !important;
    font-size: 25px;
}

/* Sidebar navigation buttons */

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background-color: #ffffff;
    color: #24527a !important;
    border: 1px solid #cbd8e6;
    border-radius: 6px;
    text-align: left;
    padding: 10px 15px;
    margin-bottom: 5px;
    font-weight: 600;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #dceaf7;
    border-color: #2878b5;
    color: #17476b !important;
}

/* Selected navigation button */

section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #2878b5;
    color: #ffffff !important;
    border-color: #2878b5;
}

/* Main buttons */

.stButton > button {
    background-color: #2878b5;
    color: #ffffff !important;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #195b8c;
    color: #ffffff !important;
}

/* Home page project overview and metrics */

[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 18px;
    border: 1px solid #dce5ef;
    border-radius: 8px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label {
    color: #263445 !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #24527a !important;
    font-weight: bold !important;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 1px solid #cbd8e6;
    border-radius: 8px;
    padding: 15px;
}

/* File uploader heading */

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] small {
    color: #263445 !important;
}

/* Browse files button */

[data-testid="stFileUploader"] button {
    color: #24527a !important;
    background-color: #eaf1f8 !important;
    border: 1px solid #b9ccdf !important;
}

/* Footer */

.footer {
    text-align: center;
    color: #52677e !important;
    font-size: 13px;
    margin-top: 35px;
    padding: 15px;
    border-top: 1px solid #dce5ef;
}

/* Mobile */

@media (max-width: 768px) {

    .main-title {
        font-size: 27px;
    }

    .subtitle {
        font-size: 14px;
    }

}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD TRAINED CNN MODEL
# ==================================================

@st.cache_resource
def load_model():

    model_path = Path("pneumonia_cnn_model.keras")
    class_path = Path("class_names.json")

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model file not found."
        )

    if not class_path.exists():
        raise FileNotFoundError(
            "Class names file not found."
        )

    model = tf.keras.models.load_model(model_path)

    with open(class_path, "r") as file:
        class_names = json.load(file)

    if len(class_names) != 2:
        raise ValueError("Expected exactly two classes.")

    return model, class_names


# ==================================================
# IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = image.resize((180, 180))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ==================================================
# HEADER
# ==================================================

def page_header():

    st.markdown(
        '<div class="main-title">'
        '🫁 Pneumonia Detection System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Deep Learning Based Chest X-Ray Classification Using CNN'
        '</div>',
        unsafe_allow_html=True
    )


# ==================================================
# FOOTER
# ==================================================

def page_footer():

    st.markdown(
        '<div class="footer">'
        'Pneumonia Detection Using CNN | BCA Final Year Project'
        '</div>',
        unsafe_allow_html=True
    )


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


st.sidebar.title("PULMO AI")

st.sidebar.write("Pneumonia Detection System")

st.sidebar.divider()


if st.sidebar.button(
    "Home",
    use_container_width=True,
    type="primary" if st.session_state.page == "Home" else "secondary"
):

    st.session_state.page = "Home"
    st.rerun()


if st.sidebar.button(
    "X-Ray Analysis",
    use_container_width=True,
    type="primary" if st.session_state.page == "X-Ray Analysis" else "secondary"
):

    st.session_state.page = "X-Ray Analysis"
    st.rerun()


if st.sidebar.button(
    "About the Model",
    use_container_width=True,
    type="primary" if st.session_state.page == "About the Model" else "secondary"
):

    st.session_state.page = "About the Model"
    st.rerun()


st.sidebar.divider()

st.sidebar.caption(
    "CNN Based Chest X-Ray Classification"
)

page = st.session_state.page


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    page_header()

    st.divider()

    st.header("Welcome to Pneumonia Detection System")

    st.write(
        "This application uses a Convolutional Neural "
        "Network (CNN) to analyze chest X-ray images "
        "and classify them into two categories: "
        "NORMAL and PNEUMONIA."
    )

    st.write("")

    left, right = st.columns(2)

    with left:

        st.subheader("About the Project")

        st.write(
            "The main objective of this project is to "
            "develop a deep learning-based system that "
            "can classify chest X-ray images using a "
            "trained CNN model."
        )

        st.write(
            "Users can upload a chest X-ray image and "
            "receive a model-generated prediction."
        )

    with right:

        st.subheader("Model Information")

        st.write("**Model:** Convolutional Neural Network")

        st.write("**Image Size:** 180 × 180 pixels")

        st.write("**Input:** Chest X-Ray Image")

        st.write("**Output:** NORMAL or PNEUMONIA")

    st.divider()

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Deep Learning", "CNN")

    with col2:
        st.metric("Image Size", "180 × 180")

    with col3:
        st.metric("Classification", "2 Classes")

    st.divider()

    st.info(
        "To analyze an X-ray image, select "
        "'X-Ray Analysis' from the sidebar."
    )

    page_footer()


# ==================================================
# X-RAY ANALYSIS PAGE
# ==================================================

elif page == "X-Ray Analysis":

    page_header()

    st.divider()

    st.header("Chest X-Ray Analysis")

    st.write(
        "Upload a chest X-ray image to analyze it "
        "using the trained CNN model."
    )

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        key="xray_upload"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.divider()

        st.subheader("Uploaded X-Ray")

        st.image(
            image,
            caption="Selected Chest X-Ray",
            width=450
        )

        analyze_button = st.button(
            "Analyze X-Ray",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            with st.spinner("Analyzing X-Ray image..."):

                try:

                    model, class_names = load_model()

                    image_array = preprocess_image(image)

                    prediction = model.predict(
                        image_array,
                        verbose=0
                    )

                    pneumonia_probability = float(
                        np.asarray(prediction).reshape(-1)[0]
                    )

                    # Class-wise probabilities

                    probabilities = {
                        class_names[0]: 1 - pneumonia_probability,
                        class_names[1]: pneumonia_probability
                    }

                    # Predicted class

                    predicted_class = max(
                        probabilities,
                        key=probabilities.get
                    )

                    confidence = probabilities[predicted_class]

                    st.divider()

                    st.subheader("Prediction Result")

                    col1, col2 = st.columns(2)

                    with col1:

                        if predicted_class == "PNEUMONIA":

                            st.error(
                                "Prediction: PNEUMONIA"
                            )

                        else:

                            st.success(
                                "Prediction: NORMAL"
                            )

                        st.metric(
                            label="Prediction Confidence",
                            value=f"{confidence * 100:.2f}%"
                        )

                    with col2:

                        st.write("### Class-wise Probability")

                        for class_name in class_names:

                            score = probabilities[class_name]

                            st.write(
                                f"**{class_name}: "
                                f"{score * 100:.2f}%**"
                            )

                            st.progress(score)

                    st.caption(
                        "These values represent model prediction "
                        "scores, not medical diagnostic certainty."
                    )

                except Exception as error:

                    st.error(
                        f"Prediction error: {error}"
                    )

    else:

        st.info(
            "Please upload a chest X-ray image to start the analysis."
        )

    st.divider()

    st.warning(
        "Medical Disclaimer: This project is developed "
        "for academic and educational purposes only. "
        "It is not a substitute for professional "
        "medical diagnosis."
    )

    page_footer()


# ==================================================
# ABOUT THE MODEL PAGE
# ==================================================

elif page == "About the Model":

    page_header()

    st.divider()

    st.header("About the CNN Model")

    st.write(
        "This project uses a Convolutional Neural Network "
        "(CNN), a deep learning architecture commonly used "
        "for image classification tasks."
    )

    st.divider()

    st.subheader("Model Specifications")

    st.write("**Model Type:** Convolutional Neural Network")

    st.write("**Input Image Size:** 180 × 180 pixels")

    st.write("**Image Format:** Grayscale")

    st.write("**Output Classes:** NORMAL and PNEUMONIA")

    st.write("**Activation Function:** ReLU and Sigmoid")

    st.write("**Loss Function:** Binary Crossentropy")

    st.write("**Optimizer:** Adam")

    st.divider()

    st.subheader("How the Model Works")

    st.write(
        "1. The user uploads a chest X-ray image."
    )

    st.write(
        "2. The image is converted into grayscale format."
    )

    st.write(
        "3. The image is resized to 180 × 180 pixels."
    )

    st.write(
        "4. The processed image is passed to the trained CNN model."
    )

    st.write(
        "5. The model generates a prediction score."
    )

    st.write(
        "6. The system displays the predicted class and confidence."
    )

    st.divider()

    st.subheader("Classification Categories")

    col1, col2 = st.columns(2)

    with col1:

        st.success("NORMAL")

        st.write(
            "The model classifies the uploaded image "
            "as belonging to the NORMAL category."
        )

    with col2:

        st.error("PNEUMONIA")

        st.write(
            "The model classifies the uploaded image "
            "as belonging to the PNEUMONIA category."
        )

    st.divider()

    st.warning(
        "This model is developed for academic and "
        "educational purposes only. Its predictions "
        "must not be considered a medical diagnosis."
    )

    page_footer()