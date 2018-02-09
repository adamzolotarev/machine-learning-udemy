# Support Vector Regression
# Explanation: http://mcminis1.github.io/blog/2014/05/10/intuition-for-SVR/
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    '~/Dropbox/github/machine_learning_udemy/machine-learning-udemy/part-2-regression/SVR/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
# y = sc_y.fit_transform(y)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))
# y.reshape(-1,1)   is needed because y   is a 1d array.
# We need it to be 1d so we can use it in regressor.fit() method.
# But StandardScaler method works with 2d arrays,
# so we are to change y  dimensions, which in Python are called shapes.
# Look at your Variable explorer and you'll see them in parenthesis,
# and the first number stands for a number of rows,
# while the second stands for a number of columns.
# We know that the number of columns should be 1.
# The number "-1" for rows commands your
# Python compiler to pick up such number of rows for the available entries
# that a matrix with one column will be formed.
# In this case it will be len(y) rows.
# np.ravel() is needed because scaling produces 2d array and we still need to use regressor.fit() . So your y  still must be 1d array, and np.ravel()  flattens any array into 1d.

# Fitting SVR to the dataset
from sklearn.svm import SVR
# rbf is for non-linear, but could also use poly
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
print(y_pred)
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
# choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
