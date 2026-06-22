import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

def load_data(path="data/abuja_real_estate.csv"):
    df = pd.read_csv(path)
    return df

def build_preprocessor():
    categorical = ['location']
    numeric = ['floor_area','bedrooms','bathrooms','year_built']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(), categorical),
        ('num', MinMaxScaler(), numeric)
    ])
    return preprocessor
