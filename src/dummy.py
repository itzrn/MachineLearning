import numpy as np
import pandas as pd

dataset = pd.read_csv("Datasets/T03_50_Startups.csv")
X_features = dataset.iloc[:, :-1].values
Y_outcome = dataset.iloc[:, [-1]].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
cT = ColumnTransformer(transformers=[
    ('OHE', OneHotEncoder(sparse=False, drop='first'),[3])
], remainder='passthrough')
X_features = cT.fit_transform(X_features)

print(X_features)
