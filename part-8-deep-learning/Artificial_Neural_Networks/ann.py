# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Before running enable tensorflow virtual environment:
# source activate tensorflow
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    '~/Dropbox/github/machine_learning_udemy/machine-learning-udemy/part-8-deep-learning/Artificial_Neural_Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
# labelencoder changes categorical value (like country) into a number

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# OneHotEncoder creates dummy variables for countries(because we have 3),
# because labelencoder translates category into numbers, but each country
# is not "higher" or "lower" than any other country
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
# Helps with performance, and we don't want one independent variable to dominate another.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Dense takes care of initializing the weights  to small numbers close to 0.
# Number of input dimensions is the number of columns in X_Test,
# it can be larger than the original dataset if you had to introduce dummy variables
# for categorical data
# output dimensions is determined through trial and error. But can use this generic rule:
# (Number of input dimensions + number of output dimensions) / 2
# Otherwise use parameter tuning
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6,  # number of nodes in the layer
                     init='uniform',  # initialize weights to small numbers close to 0
                     activation='relu',  # activation function - rectifier function
                     input_dim=11))  # number of input parameters

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN
# 'adam' is stochastic descent algorithm
# loss - the loss function to find optimal weights (e.g., in linear regression it was sum(y - y')^2 -> min)
#   if the output was more than 1 category, then we would use "categorical_crossentropy"
# metrics is the criterion to evaluate the model
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
# batch size is the number of observations after which to update the weights
# An epoch is an iteration over the entire `x` and `y`
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
# The console will show accuracy after each epoch

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# .5 is our threshold, but we could have a higher value
y_pred = (y_pred > 0.5)
print(y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
