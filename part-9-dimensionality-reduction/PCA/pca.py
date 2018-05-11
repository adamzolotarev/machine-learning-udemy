# PCA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# Importing the dataset

current_dir = os.path.dirname(os.path.realpath(__file__))
dataset = pd.read_csv(current_dir + '/Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data_scaled_train = sc.fit_transform(X_train)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
# n_components - number of extracted features that will explain
# the variance most
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Shows expained variance for each extracted featurE:
# [0.36884109 0.19318394] - so top 2 explain .36 + .19 = ~55% of variance
explained_variance = pca.explained_variance_ratio_
print("Explained variance: ", explained_variance)

# Show which variables were extracted:
# https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
print(pd.DataFrame(pca.components_, index=[
      'PC1', 'PC2'], columns=list(dataset)[:-1]).T)
# xplained variance:  [0.36884109 0.19318394]
#                            PC1       PC2
# Alcohol               0.129600 -0.498073
# Malic_Acid           -0.244641 -0.231685
# Ash                  -0.010189 -0.314969
# Ash_Alcanity         -0.240516  0.023218
# Magnesium             0.126495 -0.258420
# Total_Phenols         0.389441 -0.100685
# Flavanoids            0.427578 -0.020980
# Nonflavanoid_Phenols -0.305057 -0.039906
# Proanthocyanins       0.307753 -0.067460
# Color_Intensity      -0.110272 -0.530871
# Hue                   0.307105  0.271617
# OD280                 0.376362  0.160712
# Proline               0.281109 -0.365473

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
