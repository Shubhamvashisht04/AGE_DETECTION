import joblib
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("age_model.pkl")

# Load the image for testing (change this path to your test image)
img_path = "imdb_crop/42/nm0000142_rm3136588800_1930-5-31_2004.jpg"
image = cv2.imread(img_path)

if image is None:
    print("Image not found.")
else:
    # Preprocess the image
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.flatten().reshape(1, -1)

    # Normalize if required
    scaler = StandardScaler()
    image = scaler.fit_transform(image)

    # Predict
    predicted_age = model.predict(image)
    print(f"Predicted Age: {predicted_age[0]}")
