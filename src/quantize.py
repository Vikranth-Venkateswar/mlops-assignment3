import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Load original model from joblib
model = joblib.load("model.joblib")

# Load data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Quantize function
def quantize_array(arr):
    min_val, max_val = arr.min(), arr.max()
    scale = 255 / (max_val - min_val) if max_val != min_val else 1
    zero_point = -min_val * scale
    q = np.round(arr * scale + zero_point).astype(np.uint8)
    return q, scale, zero_point

# Quantize weights and bias
weights = model.coef_
intercept = model.intercept_

q_weights, scale_w, zp_w = quantize_array(weights)
q_intercept, scale_i, zp_i = quantize_array(np.array([intercept]))

# Define quantized model in PyTorch
class QuantizedLinearModel(nn.Module):
    def __init__(self, weights, intercept, scale_w, zp_w, scale_i, zp_i):
        super().__init__()
        self.linear = nn.Linear(len(weights), 1)
        # Dequantize weights
        deq_weights = (weights.astype(np.float32) - zp_w) / scale_w
        deq_intercept = (intercept.astype(np.float32) - zp_i) / scale_i

        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([deq_weights], dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor(deq_intercept, dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)

# Initialize and evaluate
model_q = QuantizedLinearModel(q_weights, q_intercept, scale_w, zp_w, scale_i, zp_i)
model_q.eval()

# Predict
with torch.no_grad():
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    preds = model_q(X_tensor).squeeze().numpy()

# Evaluate
r2 = r2_score(y_test, preds)
print(f"Quantized Model R² Score: {r2:.2f}")
print(f"Original Model Size: {round(model.__sizeof__() / 1024, 2)} KB")
print(f"Quantized Model Size: {round(q_weights.nbytes + q_intercept.nbytes, 2)} Bytes")
