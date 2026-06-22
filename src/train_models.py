import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from preprocessing import load_data, build_preprocessor

df = load_data()
X = df[['floor_area','bedrooms','bathrooms','location','year_built']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = build_preprocessor()

# Linear Regression
lin_model = Pipeline([('prep', preprocessor), ('lr', LinearRegression())])
lin_model.fit(X_train, y_train)
joblib.dump(lin_model, "results/linear_model.pkl")

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
poly_model = Pipeline([('prep', preprocessor), ('poly', poly), ('lr', LinearRegression())])
poly_model.fit(X_train, y_train)
joblib.dump(poly_model, "results/poly_model.pkl")

# Random Forest
rf_model = Pipeline([('prep', preprocessor), ('rf', RandomForestRegressor(n_estimators=100, random_state=42))])
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "results/rf_model.pkl")
