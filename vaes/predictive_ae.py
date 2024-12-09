import torch
import torch.nn as nn
import torch.nn.functional as F
from vaes.submodules.cnn import ConvEncoder, ConvDecoder
from vaes.submodules.mlp import MLPEncoder, MLPDecoder, MLPBottleneck, MLPActionEncoder

from functional import interact

class PredictiveAE(nn.Module):
    def __init__(self, cnn:bool=False, z_dim:int=10, **kwargs):
        super().__init__()
        assert z_dim > 0, "z_dim must be greater than 0"
        self.z_dim = z_dim
        self.encoder = ConvEncoder(z_dim) if cnn else MLPEncoder(z_dim)
        self.decoder = ConvDecoder(z_dim) if cnn else MLPDecoder(z_dim)
        self.action_encoder = MLPActionEncoder()
        self.bottleneck = MLPBottleneck(z_dim)

    def rsample(self):
        raise NotImplementedError("This method is not used by AE")
    
    # Infer the latent variables
    def infer(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        return self.encoder(x)
    
    def embed(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        h = self.encoder(x)
        z_no_act = self.action_encoder(torch.zeros(1, 5, device=x.device)).repeat(h.shape[0], 1)
        return self.bottleneck(torch.cat([h, z_no_act], dim=1))
    
    # Decode the latent variables, p(x|z)
    def decode(self, z: torch.Tensor):
        # z: (batch_size, z_dim)
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        transformed, actions = interact(x)
        h = self.infer(x)
        z_act = self.action_encoder(actions)
        z = self.bottleneck(torch.cat([h, z_act], dim=1))
        x_hat = self.decode(z)
        return x_hat, z, transformed

    def recon_loss(self, transformed: torch.Tensor, x_hat: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        # x_hat: (batch_size, 1, 28, 28)
        return F.mse_loss(x_hat, transformed, reduction='none').sum(dim=[1, 2, 3])
    
    def loss(self, _, out: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x_hat, _, transformed = out
        recon_loss = self.recon_loss(x_hat, transformed).mean()
        return recon_loss, recon_loss, None