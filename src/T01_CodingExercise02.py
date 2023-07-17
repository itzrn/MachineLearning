"""
1. Import the necessary Python libraries for data preprocessing,
including the `SimpleImputer` class from the scikit-learn library.

2. Download the "Pima Indians Diabetes Database" dataset from the
UCI Machine Learning Repository or use a preloaded version of this dataset in your local environment.

3. Load the dataset into a Pandas DataFrame using the `read_csv` function from the Pandas library.

4. Identify missing data in your dataset.
Print out the number of missing entries in each column.
Analyze its potential impact on machine learning model training.
This step is crucial as missing data can lead to inaccurate and misleading results.

5. Implement a strategy for handling missing data,
which is to replace it with the mean value, based on the nature of your dataset.
Other strategies might include dropping the rows or columns with missing data,
or replacing the missing data with a median or a constant value.

6. Configure an instance of the `SimpleImputer` class to replace missing values with the mean value of the column.

7. Apply the `fit` method of the `SimpleImputer` class on the numerical columns of your matrix of features.

8. Use the `transform` method of the `SimpleImputer` class to replace missing data in the specified numerical columns.

9. Update the matrix of features by assigning the result of the `transform` method to the correct columns.

10. Print your updated matrix of features to verify the success of the missing data replacement.
"""

# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = pd.read_csv("Datasets/pima_indians_diabetes.csv")

# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isnull().sum()
print("Missing Data \n", missing_data)

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(dataset)

# Apply the transform to the DataFrame
dataset_transformed = imputer.transform(dataset)

# printing the dataset to check we have have replace the nan value in dataset
print("Updated matrix of features: \n", dataset_transformed)
