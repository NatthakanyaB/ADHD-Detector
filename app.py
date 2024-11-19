import streamlit as st
from fastai.vision.all import *
from PIL import Image

# List of classes
all_classes = ["ADHD", "Normal"]

sample_imagesaxial = {
    "Person No.1 ADHD Axial.PNG": "sample/Person No.1 ADHD Axial.PNG",
    "Person No.2 Normal Axial.png": "sample/Person No.2 Normal Axial.png",
}
sample_imagessagittal = {
    "Person No.1 ADHD Sagittal.PNG": "sample/Person No.1 ADHD Sagittal.PNG",
    "Person No.2 Normal Sagittal.png": "sample/Person No.2 Normal Sagittal.png",
}
sample_imagescoronal = {
    "Person No.1 ADHD Coronal.PNG": "sample/Person No.1 ADHD Coronal.PNG",
    "Person No.2 Normal Coronal.png": "sample/Person No.2 Normal Coronal.png",
}

# Function to load the model
@st.cache_resource
def load_model(model_path):
    """Load a Fastai model from the given path."""
    try:
        model = load_learner(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

@st.cache_resource
def load_modelaxial():
    return load_model('Axialdetect_model_.pkl')

@st.cache_resource
def load_modelcoronal():
    return load_model('Coronaldetect_model_.pkl')

@st.cache_resource
def load_modelsagittal():
    return load_model('Sagittaldetect_model_.pkl')

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

    # Predictions
    prediction_axial = prediction_coronal = prediction_sagittal = None

    if imageaxial is not None:
        image_tensoraxial = preprocess_image(imageaxial)
        prediction_axial, _, _ = modelaxial.predict(image_tensoraxial)

    if imagecoronal is not None:
        image_tensorcoronal = preprocess_image(imagecoronal)
        prediction_coronal, _, _ = modelcoronal.predict(image_tensorcoronal)

    if imagesagittal is not None:
        image_tensorsagittal = preprocess_image(imagesagittal)
        prediction_sagittal, _, _ = modelsagittal.predict(image_tensorsagittal)

    # Display predictions
    if prediction_axial is not None:
        st.subheader("Axial Prediction:")
        st.write(f"{prediction_axial}")

    if prediction_coronal is not None:
        st.subheader("Coronal Prediction:")
        st.write(f" {prediction_coronal}")

    if prediction_sagittal is not None:
        st.subheader("Sagittal Prediction:")
        st.write(f"{prediction_sagittal}")

    # Perform hard voting
    predictions = [prediction for prediction in [prediction_axial, prediction_coronal, prediction_sagittal] if prediction is not None]
    
    if predictions:
        adhd_count = predictions.count("ADHD")
        normal_count = predictions.count("Normal")

        if adhd_count > normal_count:
            final_prediction = "ADHD"
        elif normal_count > adhd_count:
            final_prediction = "Normal"

        st.subheader("Final Prediction:")
        st.write(f"The final prediction is : {final_prediction}")

    st.subheader("Credits")
    st.write("By : Natthakanya Bhummichitra | AI-BuildersXDarunsikkhalai")
    st.markdown("Source: [GitHub](https://github.com/NatthakanyaB/ADHD-Detector)")

if __name__ == "__main__":
    main()
