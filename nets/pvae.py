import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.modules.cnn import ConvEncoder, ConvDecoder
from nets.modules.mlp import MLPEncoder, MLPDecoder
from functional import softclamp_upper, softclamp

# Poisson distribution with temperature and clamping
class Poisson:
    def __init__(self, log_rates: torch.Tensor, n_exp: int, t: float, clamp: float = 5.3):
        self.n_exp = n_exp
        self.t = t

        eps = torch.finfo(log_rates.dtype).eps
        log_rates = softclamp_upper(log_rates, clamp)
        rates = torch.exp(log_rates) + eps
        self.exp = torch.distributions.Exponential(rates)
    
    def rsample(self):
        x = self.exp.rsample((self.n_exp,)) # inter-event times
        times = torch.cumsum(x, dim=0) # arrival times of events

        indicator = times < 1.0
        z_hard = indicator.sum(0).float()

        if self.t > 0.0:
            indicator = torch.sigmoid((1-times) / self.t)
            z = indicator.sum(0)
        else:
            z = z_hard
        
        return z

class PoissonVAE(nn.Module):
    def __init__(self, cnn:bool=False, z_dim:int=10, exc_only:bool=True, **kwargs):
        super().__init__()
        assert z_dim > 0, "z_dim must be greater than 0"
        self.z_dim = z_dim
        self.encoder = ConvEncoder(z_dim=z_dim) if cnn else MLPEncoder(z_dim=z_dim) # encoder outputs to z_dim features (not z_dim*2)
        self.decoder = ConvDecoder(z_dim=z_dim) if cnn else MLPDecoder(z_dim=z_dim)
        self.register_buffer('beta', torch.tensor(1.0)) # Beta, starts at 1.0 then is reduced during training

        self.log_r = nn.Parameter((torch.rand(z_dim) * 4.0) - 6.0) # Initialize log_r, the log prior firing rate of each feature, as a trainable vector, sampled from Uniform [-6, -2]
        self.register_buffer('temp', torch.tensor(1.0)) # Temperature, starts at 1.0 then is reduced during training
        self.register_buffer('n_exp', torch.tensor(263)) # Number of samples to draw from Exponential distribution
        self.exc_only = exc_only

    def set_beta(self, beta: float):
        assert beta >= 0.0, "Beta must be greater than or equal to 0.0"
        self.beta.data.fill_(beta)

    def set_temp(self, temp: float):
        assert temp > 0.0, "Temperature must be greater than 0.0"
        self.temp.data.fill_(temp)
    
    def set_n_exp(self, n_exp: int):
        assert n_exp > 0, "Number of samples must be greater than 0"
        self.n_exp.data.fill_(n_exp)
    
    # Infer the parameters of the Poisson distributions, p(z|x)
    def infer(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        log_r = self.log_r.expand(x.size(0), -1)

        log_dr = self.encoder(x)
        if self.exc_only:
            log_dr = softclamp(log_dr, upper=10.0, lower=0.0)
        else:
            log_dr = softclamp_upper(log_dr, upper=10.0)

        # Create Poisson distribution with temperature and clamping, p(z|x)
        dist = Poisson(log_r + log_dr, self.n_exp, self.temp)

        # Also return log_dr for the KL divergence, and for downstream use (e.g. classification)
        return dist, log_dr

    
    # Poisson Reparameterization Sampling, z ~ p(z|x)
    def rsample(self, dist: Poisson):
        return dist.rsample()
    
    # Decode the latent variables, p(x|z)
    def decode(self, z: torch.Tensor):
        # z: (batch_size, z_dim)
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        dist, log_dr = self.infer(x)
        z = self.rsample(dist)
        x_hat = self.decode(z)
        return x_hat, log_dr

    def recon_loss(self, x: torch.Tensor, x_hat: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        # x_hat: (batch_size, 1, 28, 28)
        return F.mse_loss(x_hat, x, reduction='none').sum(dim=[1, 2, 3])
    
    def kl_loss(self, log_dr: torch.Tensor):
        # log_dr: (batch_size, z_dim)
        log_r = self.log_r.expand(log_dr.size(0), -1)
        f = 1 + torch.exp(log_dr) * (log_dr - 1)
        kl = torch.exp(log_r) * f
        return kl.sum(dim=1)

    def loss(self, x:torch.Tensor, out: tuple[torch.Tensor, torch.Tensor]):
        x_hat, log_dr = out
        recon_loss, kl_loss = self.recon_loss(x_hat, x).mean(), self.kl_loss(log_dr).mean()
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss