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

class MLPActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class MLPBottleneck(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim+128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int=1024, num_hidden: int=3, actv_fn: nn.Module=nn.ReLU()):
        super().__init__()
        layers = []
        in_features = in_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(actv_fn)
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)