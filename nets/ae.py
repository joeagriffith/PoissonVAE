import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.modules.cnn import ConvEncoder, ConvDecoder
from nets.modules.mlp import MLPEncoder, MLPDecoder

class AE(nn.Module):
    def __init__(self, cnn:bool=False, z_dim:int=10, **kwargs):
        super().__init__()
        assert z_dim > 0, "z_dim must be greater than 0"
        self.z_dim = z_dim
        self.encoder = ConvEncoder(z_dim=z_dim) if cnn else MLPEncoder(z_dim=z_dim)
        self.decoder = ConvDecoder(z_dim=z_dim) if cnn else MLPDecoder(z_dim=z_dim)
    
    def rsample(self):
        raise NotImplementedError("This method is not used by AE")
    
    # Infer the latent variables
    def infer(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        return self.encoder(x)
    
    # Decode the latent variables, p(x|z)
    def decode(self, z: torch.Tensor):
        # z: (batch_size, z_dim)
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        z = self.infer(x)
        x_hat = self.decode(z)
        return x_hat, z

    def recon_loss(self, x: torch.Tensor, x_hat: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        # x_hat: (batch_size, 1, 28, 28)
        return F.mse_loss(x_hat, x, reduction='none').sum(dim=[1, 2, 3])
    
    def kl_loss(self):
        raise NotImplementedError("This method is not used by AE")
    
    def loss(self, x:torch.Tensor, out: tuple[torch.Tensor, torch.Tensor]):
        x_hat, _ = out
        recon_loss = self.recon_loss(x_hat, x).mean()
        return recon_loss, recon_loss, None