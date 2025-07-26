from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import joblib

model_path = "model.joblib"
print("Loading model...")
model = joblib.load(model_path)

print("Loading dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Making predictions on test set...")
predictions = model.predict(X_test[:5])

print("Sample Predictions:")
for i, pred in enumerate(predictions):
    print(f"  House {i+1}: Predicted price = {pred:.2f}")
