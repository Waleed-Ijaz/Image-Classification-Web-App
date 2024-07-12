import streamlit as st
import os
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your Hugging Face token as an environment variable or directly in the code
HUGGINGFACE_TOKEN = os.getenv('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')

# Login to Hugging Face using the token
login(token=HUGGINGFACE_TOKEN)

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

@st.cache(allow_output_mutation=True)
def load_model():
    # Download the model from Hugging Face Hub
    model_path = hf_hub_download(repo_id="Waleed-Ijaz/dental_treatment_model", filename="Dental_CNN_v3_model.h5", use_auth_token=HUGGINGFACE_TOKEN)
    # Load the model
    model = tf.keras.models.load_model(model_path)
    return model



# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
if source_img:
    with col1:
        # Opening the uploaded image
        uploaded_image = Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Load the model
    try:
        model = load_model()
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        st.stop()

    # Classify image when button is clicked
    if st.sidebar.button('Classify Image'):
        # Preprocess the image
        img_array = np.array(uploaded_image)
        img_array = tf.image.resize(img_array, [256,256])  # Resize to the expected input size
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions)  # Get the highest confidence score

        # Check if confidence is above the selected threshold
      
        classification_result = f"Predicted Class: {predicted_class} with confidence {confidence_score:.2f}"

        with col2:
            st.image(uploaded_image, caption='Classified Image', use_column_width=True)

        with st.expander("Classification Results"):
            st.write(classification_result)
else:
    st.write("No image is uploaded yet!")
