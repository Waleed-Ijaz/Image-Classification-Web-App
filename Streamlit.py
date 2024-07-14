import streamlit as st
import numpy as np
import os
import requests
from PIL import Image

# Set your Hugging Face token directly in the code
HUGGINGFACE_TOKEN = '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'

# Setting page layout
st.set_page_config(
    page_title="Image Classification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader("Upload an image...", type=("jpg", "jpeg", "png"))
    # Model Options

# Creating main page heading
st.title("Dental Treatment Classifier")
st.caption('Upload an Image to classify')

# Define the function to make the API call
def get_prediction(image_bytes):
    model_id = "Waleed-Ijaz/dental_treatment_model"
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}"
    }
    
    response = requests.post(api_url, headers=headers, data=image_bytes)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.json()}

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
if source_img:
    with col1:
        # Opening the uploaded image
        uploaded_image = Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Classify image when button is clicked
    if st.sidebar.button('Classify Image'):
        # Convert the image to bytes
        img_bytes = source_img.read()
        
        # Get predictions from the inference API
        predictions = get_prediction(img_bytes)
        
        if "error" in predictions:
            st.error(f"Error: {predictions['message']}")
        else:
            # Process the predictions to extract class and confidence
            predicted_class = predictions['label']
            confidence_score = predictions['score']
            classification_result = f"Predicted Class: {predicted_class} with confidence {confidence_score:.2f}"
            
            with col2:
                st.image(uploaded_image, caption='Classified Image', use_column_width=True)
            
            with st.expander("Classification Results"):
                st.write(classification_result)
else:
    st.write("No image is uploaded yet!")
