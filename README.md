# MLOps Assignment 3 – End-to-End MLOps Pipeline

## Contributors
- **Name:** Vikranth Venkateswar
- **Roll Number:** [Your Roll Number]
- **Email:** [Your Email]

---

## Project Overview

This project demonstrates a complete MLOps pipeline:
- Train a **Linear Regression model** using scikit-learn.
- Containerize the model using **Docker**.
- Set up **CI/CD with GitHub Actions** to automate training, container testing, and deployment.
- Perform **manual model quantization** and analyze performance.

---

## Branch Structure

| Branch         | Purpose                                                    |
|----------------|------------------------------------------------------------|
| `main`         | Base setup (README, gitignore, file structure)             |
| `dev`          | Contains model training script `train.py`                  |
| `docker_ci`    | Contains `Dockerfile`, `predict.py`, and CI pipeline       |
| `quantization` | Contains `quantize.py` and quantized model analysis        |

---

## Dataset & Model

- **Dataset**: California Housing dataset from `sklearn.datasets`
- **Model**: Linear Regression (Scikit-learn)

---

## Docker Instructions

```bash
docker build -t mlops-assignment3 .
docker run --rm mlops-assignment3

## CI/CD Workflow

CI pipeline is configured in `.github/workflows/ci.yml`:

**Triggered on push to `docker_ci` branch**

### Steps:
-  Checkout repo
-  Install Python & dependencies
-  Run training
-  Build Docker image
-  Run container to verify `predict.py`

---

## Quantization Analysis

Manual quantization was done using **uint8 scaling** without PyTorch's built-in tools.

### Model Comparison Table

| Metric           | Original Sklearn Model | Quantized Model      |
|------------------|------------------------|-----------------------|
| R² Score         | 0.62                   | 0.60                 |
| Model Size (KB)  | 9.3 KB                 | 2.1 KB               |

---

## How to Run

### Training
```bash
python src/train.py

### Prediction
```bash
python src/predict.py

### Quantization
```bash
python src/quantize.py