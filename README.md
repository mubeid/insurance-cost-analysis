# Medical Insurance Cost Analysis

A data science project analyzing the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) to understand and predict individual medical insurance charges.

## Contents
- `data.csv` — raw dataset
- `analysis.ipynb` — full analysis: EDA, regression modeling (Linear, Polynomial, Ridge), and hyperparameter tuning
- `chat.txt` — AI chat documentation used during this project

## Key Findings
- Smoking status is the strongest predictor of insurance charges (smokers pay ~4x more on average than non-smokers)
- A strong interaction exists between BMI and smoking status: charges rise sharply for smokers once BMI exceeds ~30
- The best model (Ridge Regression with polynomial features, alpha=1, chosen via grid search) achieves R² ≈ 0.88 on the test set

## Tools
Python, pandas, scikit-learn, seaborn, matplotlib
