import torch
import torch.nn as nn
import torch.nn.functional as F
from vaes.submodules.cnn import ConvEncoder, ConvDecoder
from vaes.submodules.mlp import MLPEncoder, MLPDecoder

class VAE(nn.Module):
    def __init__(self, cnn:bool=False, z_dim:int=10, **kwargs):
        super().__init__()
        assert z_dim > 0, "z_dim must be greater than 0"
        self.z_dim = z_dim
        self.encoder = ConvEncoder(z_dim=z_dim*2) if cnn else MLPEncoder(z_dim=z_dim*2) # Encoder outputs to z_dim*2 features, for mu and logvar
        self.decoder = ConvDecoder(z_dim=z_dim) if cnn else MLPDecoder(z_dim=z_dim)
        self.register_buffer('beta', torch.tensor(1.0)) # Beta, starts at 1.0 then is reduced during training

    def set_beta(self, beta: float):
        assert beta >= 0.0, "Beta must be greater than or equal to 0.0"
        self.beta.data.fill_(beta)
    
    # Infer the parameters of the Gaussian distributions, p(z|x)
    def infer(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        h = self.encoder(x) # e.g. h[0,:] = [mu0, mu1, ..., mu_zdim, logvar0, logvar1, ..., logvar_zdim]
        mu, logvar = h.chunk(2, dim=1) # separate mu and logvar by splitting feature dimension
        return mu, logvar
    
    def embed(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        return self.encoder(x).chunk(2, dim=1)[0]
    
    # OG VAE Reparameterization trick, z ~ p(z|x)
    def rsample(self, mu: torch.Tensor, logvar: torch.Tensor):
        # mu: (batch_size, z_dim)
        # logvar: (batch_size, z_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # Decode the latent variables, p(x|z)
    def decode(self, z: torch.Tensor):
        # z: (batch_size, z_dim)
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        mu, logvar = self.infer(x)
        z = self.rsample(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def recon_loss(self, x: torch.Tensor, x_hat: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        # x_hat: (batch_size, 1, 28, 28)
        return F.mse_loss(x_hat, x, reduction='none').sum(dim=[1, 2, 3])
    
    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor):
        # mu: (batch_size, z_dim)
        # logvar: (batch_size, z_dim)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def loss(self, x:torch.Tensor, out: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x_hat, mu, logvar = out
        recon_loss, kl_loss = self.recon_loss(x_hat, x).mean(), self.kl_loss(mu, logvar).mean()
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss