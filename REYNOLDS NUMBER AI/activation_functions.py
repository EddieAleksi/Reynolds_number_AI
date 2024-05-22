import numpy as np

def mlp_forward(X):
    # Initializing weights for each layer
    theta1 = np.random.random([5, 5])
    theta2 = np.random.random([5, 5])
    theta3 = np.random.random([5, 2])

    # Forward propagation through each layer
    h1 = np.tanh(X @ theta1)  # First hidden layer
    h2 = np.tanh(h1 @ theta2)  # Second hidden layer
    h3 = np.tanh(h2 @ theta3)  # Output layer

    return h3

# Generate random input
X = np.random.random((10000, 5))

# Test the MLP model
output = mlp_forward(X)
print("Input shape:", X.shape)
print("Output shape:", output.shape)
