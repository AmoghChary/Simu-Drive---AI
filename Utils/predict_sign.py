import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/traffic_sign_model.h5")
IMG_SIZE = 48
CLASS_NAMES = [f"Class {i}" for i in range(43)]  # Replace with real names if needed

# Predict function
def predict_sign(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_id = np.argmax(pred)
    confidence = np.max(pred)
    return CLASS_NAMES[class_id], confidence

