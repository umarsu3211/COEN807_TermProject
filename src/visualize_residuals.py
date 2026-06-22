import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from preprocessing import load_data

# Load dataset
df = load_data()
X = df[['floor_area','bedrooms','bathrooms','location','year_built']]
y = df['price']

# Load trained models
lin_model = joblib.load("results/linear_model.pkl")
poly_model = joblib.load("results/poly_model.pkl")
rf_model = joblib.load("results/rf_model.pkl")

# Predictions
y_pred_lin = lin_model.predict(X)
y_pred_poly = poly_model.predict(X)
y_pred_rf = rf_model.predict(X)

# Residuals
res_lin = y - y_pred_lin
res_poly = y - y_pred_poly
res_rf = y - y_pred_rf

# Plot residuals
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.scatterplot(x=y_pred_lin, y=res_lin)
plt.axhline(0, color='red', linestyle='--')
plt.title("Linear Regression Residuals")

plt.subplot(1,3,2)
sns.scatterplot(x=y_pred_poly, y=res_poly)
plt.axhline(0, color='red', linestyle='--')
plt.title("Polynomial Regression Residuals")

plt.subplot(1,3,3)
sns.scatterplot(x=y_pred_rf, y=res_rf)
plt.axhline(0, color='red', linestyle='--')
plt.title("Random Forest Residuals")

plt.tight_layout()
plt.savefig("results/residual_plots.png")
plt.show()
