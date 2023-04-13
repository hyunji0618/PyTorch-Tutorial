import torch
import numpy as np

# empty: create tensor with uninitialized values
x = torch.empty(1) # tensor([7.3787e+22])

x = torch.empty(3) # tensor([-2.0000e+00,  1.4020e+03,  1.0895e+10])

# 2D tensor
x = torch.empty(2, 3) # tensor([[5.1163e-14, 7.9463e+08, 1.1210e-44],[3.6734e-40, 6.4038e-30, 1.4013e-45]])

# 4D tensor
x = torch.empty(2, 2, 2, 3)

# rand: tensor with random values
x = torch.rand(2) # tensor([0.2104, 0.0021])

x = torch.zeros(2) # tensor([0., 0.])
x = torch.ones(3) # tensor([1., 1., 1.])

x = torch.ones(2, dtype=torch.int) # tensor([1, 1], dtype=torch.int32)
x = torch.ones(2, dtype=torch.double) # tensor([1., 1.], dtype=torch.float64)

x = torch.tensor([2.5, 0.1]) # tensor([2.5000, 0.1000])

# addition
x = torch.rand(2, 2) # tensor([[0.2632, 0.9957], [0.0754, 0.0671]])
y = torch.rand(2, 2) # tensor([[0.6636, 0.3769], [0.6593, 0.9422]])
z = x + y # tensor([[1.6768, 1.1591], [1.0484, 1.4184]])
z = torch.add(x, y) # tensor([[1.6768, 1.1591], [1.0484, 1.4184]])
y.add_(x) # y = x + y

# subtraction
z = x - y
z = torch.sub(x, y)
y.sub(x)

# multiplication
z = x * y
z = torch.mul(x, y)
y.mul(x)

# division
z = x / y
z = torch.div(x, y)
y.div(x)

x = torch.rand(2, 3)
# print(x) -> tensor([[0.5419, 0.1173, 0.8990], [0.3989, 0.9065, 0.0574]])
# print(x[:, 0]) -> tensor([0.5419, 0.3989])
# print(x[1, :]) -> tensor([[0.3989, 0.9065, 0.0574])
# print(x[1, 1]) -> tensor(0.9065)
# print(x[1, 1].item()) -> 0.9065
y = x.view(6) # tensor([0.5419, 0.1173, 0.8990, 0.3989, 0.9065, 0.0574])
# automatically designs the dimension
y = x.view(-1, 3) # tensor([[0.5419, 0.1173, 0.8990], [0.3989, 0.9065, 0.0574]])

# import numpy to torch
a = torch.ones(5)
b = a.numpy() # [1. 1. 1. 1. 1.]
print(type(b)) # <class 'numpy.ndarray'>

# imply torch to numpy array
a = np.ones(5)
b = torch.from_numpy(a)
print(b) # tensor([1., 1., 1., 1., 1.], dtype=torch.float64)