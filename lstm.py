import torch.nn as nn

lstm_net = nn.LSTM(input_size=128, hidden_size=256, bias=False)
num_params = sum(p.numel() for p in lstm_net.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params}")
print(f"Dimensionality of cell state: {lstm_net.hidden_size}")

