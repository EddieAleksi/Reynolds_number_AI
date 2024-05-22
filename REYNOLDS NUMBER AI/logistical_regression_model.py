import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Data set for training
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 1, 1])  # Adjusted to match the number of samples in X

# Initialize and train the model
model = LogisticRegression(penalty=None)  # Updated penalty argument
model.fit(X, y)  # Proper method call to train the model

# Making predictions
yhat = model.predict(X)
print("Predictions:", yhat)

# Plotting the results
plt.scatter(X[:, 0], X[:, 1], c=yhat, cmap=plt.cm.coolwarm, s=500)  # Corrected colormap access
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()  # This line will display the plot
