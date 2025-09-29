import torch
inputs = torch.rand(100,3)
weights = torch.tensor([[1.1],[2.2],[3.3]])
bias = torch.tensor(4,4)
targets = input @ weights + bias + 0.1 * torch.randn(100,1)