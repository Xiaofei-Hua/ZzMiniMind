import torch

x = torch.tensor([1, 2, 3, 4, 5])
mask = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
print(x[mask])

