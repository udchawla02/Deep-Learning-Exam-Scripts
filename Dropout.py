import numpy as np

# Given values
W = np.array([[1, 2],
              [1, 2]])
x = np.array([-1, 2])
dropout_mask = np.array([1, 0])
dropout_rate = 0.5
scaling_factor = 1 / (1 - dropout_rate)

# Reshape x as a column vector (2x1 matrix)
x_reshaped = x.reshape(-1, 1)
print(x_reshaped)
# Linear transformation
y = np.matmul(W,x_reshaped)

print(y)
# Apply dropout mask and scaling factor
output = dropout_mask * (scaling_factor * y)

print("Output:", output)