import tensorflow as tf
import streamlit as st
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Set page title
st.title("Fruit Detection In Image")
st.write("Upload an image containing a fruit to get the prediction.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
class_labels = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'stawberries', 'watermelon', 'pineapple']

if uploaded_image is not None:
    image = load_img(uploaded_image, target_size=(224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize pixel values if needed
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Eurther processing
    if st.button("Predict"):
        model = tf.keras.models.load_model("best_fruit_model.h5")
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        st.header("Predicted class:")
        st.header(predicted_label)
