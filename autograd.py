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
