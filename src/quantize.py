import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Load model trained with train.py
model = joblib.load("model.joblib")

# Load dataset
X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Quantize weights and bias to float16
weights = model.coef_.astype(np.float16)
intercept = np.array([model.intercept_], dtype=np.float16)

# Define quantized model
class QuantizedLinearModel(nn.Module):
    def __init__(self, weights, intercept):
        super().__init__()
        self.linear = nn.Linear(len(weights), 1)
        self.linear.weight = nn.Parameter(torch.tensor([weights], dtype=torch.float32))
        self.linear.bias = nn.Parameter(torch.tensor(intercept, dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)

# Evaluate
model_q = QuantizedLinearModel(weights, intercept)
model_q.eval()

with torch.no_grad():
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    preds = model_q(X_tensor).squeeze().numpy()

r2 = r2_score(y_test, preds)
print(f"Quantized Model R² Score: {r2:.4f}")
print(f"Original Model Size: {round(model.__sizeof__() / 1024, 2)} KB")
print(f"Quantized Model Size: {weights.nbytes + intercept.nbytes} Bytes")
