# 😓 Stress Detection with Machine Learning

This project implements a machine learning model to detect stress levels based on physiological signals such as Electrodermal Activity (EDA), skin temperature, and heart rate.

## 🎯 Goal
To build an accurate and interpretable classifier that identifies whether a subject is under stress based on sensor input data.

## 📁 Dataset
The dataset includes physiological signals collected from multiple subjects. Each data point includes:
- `EDA` – Electrodermal activity
- `Temp` – Skin temperature
- `Label` – Session type (1 = task session, 0 = rest)
- `Subject` – Subject identifier
- `is_stress` – Target label (1 = stressed, 0 = not stressed)

## 🤖 Model
The model used is a gradient boosting classifier (XGBoost), trained on a processed version of the dataset. Feature scaling and label encoding were applied.

## 📈 Results
- **Accuracy:** ~92.7%
- **Evaluation:** Predictions were logged on a validation set of 1000 samples and saved in `predictions_log.csv`.

## 🛠️ Project Structure
