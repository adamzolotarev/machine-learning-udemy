# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
# axis 0 means mean along columns; axis 1 - across rows.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean' , axis = 0)
imputer = imputer.fit(X[:, 1:3])
# take all columns; fix column 1 and 2 (upper is ecxluded)
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)