import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Initializing the model with specific parameters
model = MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu', max_iter=1000, random_state=0)

# XOR dataset for training
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Fitting the model
model.fit(X, y)

# Making predictions
yhat = model.predict(X)

# Plotting predictions
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=yhat, cmap=plt.cm.coolwarm, s=500)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Predictions for XOR dataset")

# Assuming plot_decision_boundary is defined to visualize the decision surface
# Function to plot decision boundaries
def plot_decision_boundary(X, y, clf):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Plotting decision boundary
plt.figure()
plot_decision_boundary(X, y, model)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision boundary for MLP classifier on XOR dataset")
plt.show()
