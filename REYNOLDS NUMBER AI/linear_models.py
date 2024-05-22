# Required packages
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import warnings

# Configuration settings
np.random.seed(0)
warnings.filterwarnings('ignore')
plt.style.use("dark_background")

# Data set for training
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Adjusted to match the number of rows in X

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=500)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
