# Data Preprocessing Template

# Importing the libraries
# numpy is for math
import numpy as np
# chart plotting
import matplotlib.pyplot as plt
# to import and manage datasets
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# print(dataset)
# take all rows, all columns except for the last
X = dataset.iloc[:, :-1].values
# take all rows, and only 3rd column
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
# cross_validation is deprecated
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# random_state set to 0 just so we have the same results as in the course
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train) -no longer usable
# y_train = np.ravel(sc_y.fit_transform(y_train.reshape(-1, 1)))

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
