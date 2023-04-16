import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2

print(loss) # tensor(1., grad_fn=<PowBackward0>)

loss.backward()
print(w.grad) # tensor(-2.)

### update weights
### next forward and backward