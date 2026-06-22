# COEN807_TermProject
Machine learning for Real-world data analyctic

# COEN807 House Price Prediction

## Overview
This project predicts residential property prices in Abuja using supervised learning models:
- Linear Regression
- Polynomial Regression
- Random Forest

## Dataset
Source: Kaggle Nigeria Real Estate Dataset (~5,000 records).
Features: floor area, bedrooms, bathrooms, location, year built.
Target: Price (in NGN millions).

## Methodology
- Preprocessing: missing value imputation, one-hot encoding, scaling.
- Models: Linear, Polynomial, Random Forest.
- Split: 70/15/15 (train/validation/test).
- Metrics: MAE, RMSE, R².

## Results
| Model              | MAE | RMSE | R²  |
|--------------------|-----|------|-----|
| Linear Regression  | 3.12| 4.33 | 0.61|
| Polynomial         | 2.85| 3.95 | 0.68|
| Random Forest      | 2.10| 3.20 | 0.82|

## Outputs
- Residual plots
- Metrics table
- Trained Random Forest model

## Reproducibility
1. Clone repo
2. Install requirements: `pip install -r requirements.txt`
3. Run pipeline: `python src/pipeline.py`

## Future Work
- Expand dataset beyond Abuja
- Fairness-aware ML
- SHAP interpretability
- ## Dataset Access Instructions

The raw dataset (`abuja_real_estate.csv`) is excluded from this repository to keep the repo lightweight (see `.gitignore`).  
To reproduce results:

1. Download the dataset from Kaggle: [Nigeria Real Estate Dataset](https://www.kaggle.com/datasets) or another credible source.
2. Save the file as `abuja_real_estate.csv`.
3. Place the file inside the `data/` folder of this project:
