import os
import joblib

# Dummy model if needed
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Replace this with actual training before saving
model.fit([[0], [1]], [0, 1])

# Safe save
output_path = os.path.join(os.getcwd(), "age_model.pkl")
joblib.dump(model, output_path)

print(f"Model saved at: {output_path}")
