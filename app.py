# -*- coding: utf-8 -*-
"""app

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10EErEMRSIv4uiLeLHB6XDNO_MdLls0OK
"""

import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pandas as pd

# List of classes
all_classes = ["ADHD", "Normal"]

sample_imagesaxial = {
    "Normal Axial sample": "Sample/Normal Axial sample.png",
    "ADHD Axial sample": "Sample/ADHD Axial sample.PNG",
}
sample_imagessagittal = {
    "Normal Sagittal sample": "Sample/Normal Sagittal sample.png",
    "ADHD Sagittal sample": "Sample/ADHD sagittal sample.PNG",
}
sample_imagescoronal = {
    "ADHD Coronal sample": "Sample/ADHD Coronal sample.PNG",
    "Normal coronal sample": "Sample/Normal coronal sample.png",
}

# Function to load the model
@st.cache_resource
def load_model(model_path):
    """Load a Fastai model from the given path."""
    try:
        # Load the Fastai model
        model = load_learner(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

@st.cache_resource
def load_modelaxial():
    return load_model('Axial_model_.pkl')

@st.cache_resource
def load_modelcoronal():
    return load_model('Coronal_model_.pkl')

@st.cache_resource
def load_modelsagittal():
    return load_model('Sagittal_model_.pkl')

# Function for preprocessing the image
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return PILImage.create(image)

# Main function
def main():
    modelaxial = load_modelaxial()
    modelcoronal = load_modelcoronal()
    modelsagittal = load_modelsagittal()

    st.title("ADHD Detector")
    st.write("Upload an image")

    uploaded_fileaxial = st.file_uploader("Choose an axial image...")
    uploaded_filecoronal = st.file_uploader("Choose a coronal image...")
    uploaded_filesagittal = st.file_uploader("Choose a sagittal image...")

    with st.expander("Or choose from sample axial here..."):
        axialsample = st.selectbox(label="Select here", options=list(sample_imagesaxial.keys()), label_visibility="hidden")
    with st.expander("Or choose from sample coronal here..."):
        coronalsample = st.selectbox(label="Select here", options=list(sample_imagescoronal.keys()), label_visibility="hidden")
    with st.expander("Or choose from sample sagittal here..."):
        sagittalsample = st.selectbox(label="Select here", options=list(sample_imagessagittal.keys()), label_visibility="hidden")

    # Initialize images
    imageaxial = imagecoronal = imagesagittal = None

    if uploaded_fileaxial is not None:
        imageaxial = Image.open(uploaded_fileaxial)
        st.image(imageaxial, caption='Uploaded Axial Image', use_column_width=True)
    elif axialsample:
        imageaxial = Image.open(sample_imagesaxial[axialsample])
        st.image(imageaxial, caption=f'Selected Sample: {axialsample}', use_column_width=True)

    if uploaded_filecoronal is not None:
        imagecoronal = Image.open(uploaded_filecoronal)
        st.image(imagecoronal, caption='Uploaded Coronal Image', use_column_width=True)
    elif coronalsample:
        imagecoronal = Image.open(sample_imagescoronal[coronalsample])
        st.image(imagecoronal, caption=f'Selected Sample: {coronalsample}', use_column_width=True)

    if uploaded_filesagittal is not None:
        imagesagittal = Image.open(uploaded_filesagittal)
        st.image(imagesagittal, caption='Uploaded Sagittal Image', use_column_width=True)
    elif sagittalsample:
        imagesagittal = Image.open(sample_imagessagittal[sagittalsample])
        st.image(imagesagittal, caption=f'Selected Sample: {sagittalsample}', use_column_width=True)

    if imageaxial is not None:
        image_tensoraxial = preprocess_image(imageaxial)
        pred_axial, pred_idx_axial, pred_probs_axial = modelaxial.predict(image_tensoraxial)

    if imagecoronal is not None:
        image_tensorcoronal = preprocess_image(imagecoronal)
        pred_coronal, pred_idx_coronal, pred_probs_coronal = modelcoronal.predict(image_tensorcoronal)

    if imagesagittal is not None:
        image_tensorsagittal = preprocess_image(imagesagittal)
        pred_sagittal, pred_idx_sagittal, pred_probs_sagittal = modelsagittal.predict(image_tensorsagittal)

    # Display predictions
    if 'pred_axial' in locals():
        st.subheader("Axial Prediction:")
        st.write(f"Predicted label: {pred_axial} with probability: {pred_probs_axial[pred_idx_axial]:.4f}")

    if 'pred_coronal' in locals():
        st.subheader("Coronal Prediction:")
        st.write(f"Predicted label: {pred_coronal} with probability: {pred_probs_coronal[pred_idx_coronal]:.4f}")

    if 'pred_sagittal' in locals():
        st.subheader("Sagittal Prediction:")
        st.write(f"Predicted label: {pred_sagittal} with probability: {pred_probs_sagittal[pred_idx_sagittal]:.4f}")

    st.subheader("Credits")
    st.write("By : Natthakanya Bhummichitra | AI-BuildersXDarunsikkhalai")
    st.markdown("Source: [GitHub](https://github.com/NatthakanyaB/ADHD-Detector)")

if __name__ == "__main__":
    main()