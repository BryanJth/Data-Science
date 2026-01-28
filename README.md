# Annual Diabetes Cases Prediction (Indonesia) — Neural Network Regression

This repository contains a small end-to-end machine learning project to **predict annual diabetes cases in Indonesia (2010–2024)** using a **feedforward Neural Network (regression)** built with **Keras/TensorFlow**.

The goal is to learn non-linear relationships between socio-demographic/economic indicators and the **total diabetes cases (in millions)**, then evaluate the model on recent years (2022–2024). 

---

## Project Summary

**Task:** Regression (predict continuous values)  
**Target (y):** Total Diabetes Cases (Millions)  
**Features (X):**
- Population (Millions)
- Average sugar consumption per week (ons)
- Population aged > 40 (Millions)
- Sugar price per 1 kg

**Train/Test Split (time-based):**
- Train: 2010–2021  
- Test: 2022–2024 

---

## Key Findings (From This Project)

### 1) Strongest correlated feature
On the training set correlation analysis:
- **Population aged > 40** has the strongest positive correlation with diabetes cases (**0.85**).
- **Total population** is also strongly positive (**0.72**).
- **Sugar price** shows weak positive correlation (**0.31**).
- **Sugar consumption** shows weak negative correlation (**−0.28**) (likely noise / proxy mismatch). 

> Note: Total population and population aged >40 are highly correlated (**0.94**), indicating potential redundancy / multicollinearity risk. :contentReference[oaicite:4]{index=4}

### 2) Test predictions (2022–2024)
| Year | Actual | Predicted |
|------|--------|-----------|
| 2022 | 19.5 | 19.298 |
| 2023 | 20.4 | 21.593 |
| 2024 | 20.0 | 19.821 | 

The model follows the overall trend, with the largest deviation in 2023 (overestimate ~1.19M). :contentReference[oaicite:6]{index=6}

### 3) Evaluation metrics (test set: 2022–2024)
- **MAE:** 0.52 (≈ ±520k cases on average)
- **MSE:** 0.50
- **R²:** −2.68 (negative indicates poor variance explanation; also affected by very small test size) 

---

## Methodology

### 1) Preprocessing
- Fix numeric parsing (decimal comma → decimal point).
- **Standardize** input features using `StandardScaler` (mean=0, std=1).
- Split train/test by year (time-based). :contentReference[oaicite:8]{index=8}

### 2) Model Architecture (Neural Network Regression)
Feedforward fully-connected model:
- Input layer: 4 neurons (4 features)
- Hidden layer 1: 10 neurons, **ReLU**
- Dropout: 0.5
- Hidden layer 2: 5 neurons, **ReLU**
- Dropout: 0.5
- Output layer: 1 neuron, **Linear** (regression output) :contentReference[oaicite:9]{index=9}

### 3) Training Setup
- Loss: **MSE**
- Optimizer: **SGD**, learning rate = 0.003
- Batch size: 1
- Max epochs: 1000 (usually stops earlier with callbacks)
- Callbacks:
  - **EarlyStopping** (patience=10, restore_best_weights=True)
  - **ReduceLROnPlateau** (patience=5, factor=0.5, min_lr=1e−10) :contentReference[oaicite:10]{index=10}

---

## Repository Structure (Current)

```text
Data-Science/
├─ Dataset.csv
├─ Makalah (1).pdf
├─ Neural Network Code.ipynb
└─ Presentation (1).pdf
