import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

dataset = pd.read_csv("Datasets/covid_toyColumnTransformation.csv")

cT = ColumnTransformer(transformers=[  # we have to pass three tuple
    ('tnf1', SimpleImputer(missing_values=np.nan, strategy='mean'), ['fever']),  # for simpleImputer
    ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']], dtype=np.int32), ['cough']),  # for ordinal Encoding
    ('tnf3', OneHotEncoder(sparse=False, drop='first', dtype=np.int32), ['gender', 'city'])   # for nominal Encoding -> OneHotEncoding
], remainder='passthrough')
"""
in above code we can do two things for remainder -> passthrough or drop
pass through means the columns at which u are not applying keep with for training and testing further
drop means the columns at which u are not applying anything drop it 
like at age column we are performing anything so it will remain there if we use passthrough 
and will get drop if we will use drop
"""
dataset = cT.fit_transform(dataset)  # numpy.ndarray
print(dataset)
