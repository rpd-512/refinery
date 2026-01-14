import torch
import torch.nn as nn

# Example vectors
y = torch.tensor([1.0, 2.0, 3.0])
y_hat = torch.tensor([0.5, 2.5, 2.0])

# 1. Mean Squared Error
mse = nn.MSELoss()
loss_value = mse(y_hat, y)  # y_hat is prediction, y is target
print("MSE Loss:", loss_value.item())

# 2. L1 / Mean Absolute Error
l1 = nn.L1Loss()
l1_value = l1(y_hat, y)
print("L1 Loss:", l1_value.item())
