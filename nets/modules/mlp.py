import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim),
        )
    
    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor):
        x = self.net(x)
        return x.view(x.size(0), 1, 28, 28)
