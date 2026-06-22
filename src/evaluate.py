import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json
from preprocessing import load_data

df = load_data()
X = df[['floor_area','bedrooms','bathrooms','location','year_built']]
y = df['price']

models = {
    "Linear Regression": "results/linear_model.pkl",
    "Polynomial Regression": "results/poly_model.pkl",
    "Random Forest": "results/rf_model.pkl"
}

results = {}
for name, path in models.items():
    model = joblib.load(path)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

with open("results/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print(results)
