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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# 1) Design model


# 2) Construct loss and optimizer


# 3) Training loop

    # forward pass and loss

    
    # backward pass
    
    # update the progress

        
# plot
