import torch

# Create a massive matrix directly on the GPU
x = torch.randn(20000, 20000, device="cuda")

# Do some heavy math
y = torch.matmul(x, x)

print("Computation complete! Check your Task Manager now.")