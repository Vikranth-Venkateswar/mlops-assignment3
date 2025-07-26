from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os

print("Loading dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"R² score on test set: {score:.4f}")

model_path = "model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
