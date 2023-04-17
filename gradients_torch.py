# Gradient Descent with Autograd and Backpropagation (using PyTorch gradient descent)

import torch

# Linear Regression: f = w * x

# f = 2x (weight = 2)
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# Initialize the weight (start with w = 0)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) 

# Model prediction (forward pass): calculate y_predicted
def forward(x):
    return w * x

# Calculate loss (MSE: mean squared)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

print(f'Prediction before the training: f(5) = {forward(5):.3f}') # f(5) = 0.000

# Start training
learning_rate = 0.01
n_iter = 100

for epoch in range(n_iter):
    y_pred = forward(X) # prediction (forward pass)
    l = loss(Y, y_pred) # loss
    l.backward() # dl/dw
    with torch.no_grad(): # set requires_grad=False
        w -= learning_rate * w.grad # update weights
    # zero gradients (use backward() -> need initialization because it changes the value automatically)
    w.grad.zero_() # gradients = 0
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after the training: f(5) = {forward(5):.3f}') # f(5) = 10.000
