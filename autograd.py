# Gradient Calculation With Autograd

import torch

x = torch.randn(3, requires_grad=True)
print(x) # tensor([ 1.1861, -0.3506, -1.6506], requires_grad=True)

y = x + 2
print(y) # tensor([3.1861, 1.6494, 0.3494], grad_fn=<AddBackward0>)

z = y * y * 2
z = z.mean() # tensor(16.6866, grad_fn=<MeanBackward0>)
print(z)

# calculating gradient of z with respect to x
z.backward() # dz/dx
print(x.grad) # tensor([3.7406, 3.6692, 4.1282])

# disable pytorch from using the gradient function
x.requires_grad_(False)
y = x.detach()
with torch.no_grad():
    y = x * 2
    
# Training Example
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()
    
# tensor([3., 3., 3., 3.])
# tensor([3., 3., 3., 3.])
# tensor([3., 3., 3., 3.]) 

# Optimization
weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD(weights, lr = 0.01)
optimizer.step()
optimizer.zero_grad()
