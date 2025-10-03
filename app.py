import streamlit as st
import numpy as np
import cv2
import joblib

# Load your trained model
model = joblib.load("digit_classifier.pkl")

st.title("📷 Handwritten Digit Recognition")
st.write("Take a photo of your handwritten digit (0-9) and I'll predict it!")

# Capture image from camera
camera_image = st.camera_input("Take a picture")

if camera_image:
    # Convert image to numpy array
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Show original image
    st.image(img, caption="Captured Image", use_column_width=True)

    # Preprocess for MNIST model (28x28, invert, normalize, flatten)
    img_resized = cv2.resize(img, (28, 28))
    img_resized = 255 - img_resized   # Invert colors (MNIST is white digit on black background)
    img_flatten = img_resized.flatten().reshape(1, -1) / 255.0

    # Predict using the trained model
    prediction = model.predict(img_flatten)[0]

    st.success(f"✅ Predicted Digit: {prediction}")

    # Show processed image
    st.write("Processed (28x28) Image:")
    st.image(img_resized, width=150, caption="Processed for Model", clamp=True)