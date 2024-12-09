import torch
import torch.nn as nn

class EncBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, pool=False,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
            nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity(),
            # nn.BatchNorm2d(out_dim),
            nn.SiLU(),
        )
    
    def forward(self, x):
        return self.net(x)

class ConvEncoder(nn.Module):
    def __init__(self, z_dim: int, bn=True):
        super().__init__()
        self.cnn = nn.Sequential(
            EncBlock(1, 32, 3, 1, 1, pool=True),
            EncBlock(32, 64, 3, 1, 1, pool=True),
            EncBlock(64, 128, 3, 1, 0),
            EncBlock(128, 256, 3, 1, 0),
            EncBlock(256, z_dim, 3, 1, 0),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z_dim, z_dim * 8),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(z_dim * 8, z_dim),
            nn.LayerNorm(z_dim)
		)
    
    def forward(self, x):
        return self.mlp(self.cnn(x))

class DecBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride),
            # nn.BatchNorm2d(out_dim),
            nn.SiLU(),
        )
    
    def forward(self, x):
        return self.net(x)
    
class ConvDecoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            DecBlock(z_dim, 512, 3, 1),
            DecBlock(512, 256, 3, 3),
            DecBlock(256, 128, 3, 3),
            DecBlock(128, 64, 2, 1),
            nn.Conv2d(64, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(2).unsqueeze(3))