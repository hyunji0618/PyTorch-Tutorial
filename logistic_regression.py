# Logistic Regression

# Pipeline
## 1) Design model (input, output size, forward pass)
## 2) Construct loss and optimizer
## 3) Training loop
##   - forward pass: compute prediction
##   - back pass: gradients
##   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets # generate regression dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare regression datasets
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

n_samples, n_features = x.shape


# 1) Design model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) Construct loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, Y)
    
    # backward pass
    loss.backward()
    
    # update the progress
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
        
# plot
predicted = model(X).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()