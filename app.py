import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model(r'C:\Users\mosta\Downloads\streamlitcv\modelll-24-0.9981.keras')
# Define class names
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to the input size your model expects
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title('Image Classification with Streamlit')
st.write("Upload an image to classify it!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]

    # Display the results
    st.write(f'Predicted Class: {predicted_class}')