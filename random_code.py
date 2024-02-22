import numpy as np
import torch
import torch.nn as nn

# try out random code

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 8),
    nn.Sigmoid()
)

# Count the number of trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {total_params}")