import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.dropna()
    scaler = StandardScaler()
    df[['feature1','feature2']] = scaler.fit_transform(df[['feature1','feature2']])
    return df
