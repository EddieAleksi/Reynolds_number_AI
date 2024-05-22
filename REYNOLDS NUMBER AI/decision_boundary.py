import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Data set for training
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Adjusted to match the number of samples in X

# Initialize and train the model
model = LogisticRegression(penalty=None)
model.fit(X, y)

# Function to plot decision boundaries
def plot_decision_boundary(X, y, clf):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Correcting spelling mistakes and method names
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # np.c_ needs to be corrected as np.c_[]
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Using plt.contourf to plot the contour
    plt.contourf(xx, yy, Z, cmap=plt.cm.bwr, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k')

# Call the function to plot the decision boundary
plot_decision_boundary(X, y, model)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")  # Corrected from second xlabel to ylabel
plt.title("Decision boundary for logistic classifier on dataset")
plt.show()

# XOR Data set for training
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # Adjusted for XOR-like output

# Creating a pipeline with Polynomial Features and Logistic Regression
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('logistic', LogisticRegression(penalty=None))
])
model.fit(X, y)

# Function to plot decision boundaries
def plot_decision_boundary(X, y, clf):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.bwr, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k')

# Call the function to plot the decision boundary
plot_decision_boundary(X, y, model)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision boundary for logistic classifier on XOR dataset")
plt.show()
