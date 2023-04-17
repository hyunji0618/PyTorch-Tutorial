# Gradient Descent with Autograd and Backpropagation 4
## Use PyTorch to compute gradients, loss, parameter updates, and model prediction
## use Autograd, PyTorch Loss Class, PyTorch Optimizer Class, PyTorch Model Class

import torch
import torch.nn as nn # neural network module

# Linear Regression: f = w * x

# f = 2x (weight = 2)
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features) # 4 1

# Initialize the weight (start with w = 0)
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Prediction before the training: f(5) = {model(X_test).item():.3f}') # f(5) = 2.688

# Start training
learning_rate = 0.01
n_iter = 100

loss = nn.MSELoss() # loss = MSE
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # SGD = stochastic gradient descent

for epoch in range(n_iter):
    y_pred = model(X) # prediction (forward pass)
    l = loss(Y, y_pred) # loss
    l.backward() # dl/dw
    optimizer.step() # update weights
    # zero gradients (use backward() -> need initialization because it changes the value automatically)
    optimizer.zero_grad() # gradients = 0
    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack
        print(f'Epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
        
print(f'Prediction after the training: f(5) = {model(X_test).item():.3f}') # f(5) = 9.637