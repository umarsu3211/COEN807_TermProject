import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from preprocessing import load_data

# Load dataset
df = load_data()
X = df[['floor_area','bedrooms','bathrooms','location','year_built']]
y = df['price']

# Load Random Forest model
rf_model = joblib.load("results/rf_model.pkl")

# Extract feature names after preprocessing
feature_names = rf_model.named_steps['prep'].get_feature_names_out()
importances = rf_model.named_steps['rf'].feature_importances_

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.show()
