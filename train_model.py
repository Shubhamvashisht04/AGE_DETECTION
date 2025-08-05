import pandas as pd
import numpy as np
import os
import cv2
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load CSV
df = pd.read_csv("processed.csv")

# Prepare features and labels
X = []
y = []

for i in range(len(df)):
    try:
        path = df.iloc[i]["path"]
        age = df.iloc[i]["age"]

        img_path = os.path.join("imdb_crop", path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img.flatten())
        y.append(age)
    except:
        continue

X = np.array(X)
y = np.array(y)

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
with open("age_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as age_model.pkl")
