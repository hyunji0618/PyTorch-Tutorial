# Gradient Descent with Autograd and Backpropagation 1
## Not using PyTorch - manually calculating everything

# Pipeline
## 1) Design model (input, output size, forward pass)
## 2) Construct loss and optimizer
## 3) Training loop
##   - forward pass: compute prediction
##   - back pass: gradients
##   - update weights

import numpy as np

# Linear Regression: f = w * x

# f = 2x (weight = 2)
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0 # Initialize the weight (start with w = 0)

# Model prediction (forward pass): calculate y_predicted
def forward(x):
    return w * x

# Calculate loss (MSE: mean squared)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# Calculate gradient
# loss (MSE) = 1/N * (w*x - y)**2
# dL/dw = 1/N * 2x(wx - y): gradient
def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()

print(f'Prediction before the training: f(5) = {forward(5):.3f}') # f(5) = 0.000

# Start training
learning_rate = 0.01
n_iter = 20

for epoch in range(n_iter):
    y_pred = forward(X) # prediction (forward pass)
    l = loss(Y, y_pred) # loss
    dw = gradient(X, Y, y_pred) # gradients
    w -= learning_rate * dw # update weights
    if epoch % 2 == 0:
        print(f'Epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after the training: f(5) = {forward(5):.3f}') # f(5) = 10.000
