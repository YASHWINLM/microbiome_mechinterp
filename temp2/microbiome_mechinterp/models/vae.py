import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Variational Autoencoder for microbiome data"""

    def __init__(
        self, input_dim, hidden_dims=[512, 256, 128], latent_dim=50, dropout_rate=0.3
    ):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

    def encode(self, x):
        """Encode input to latent parameters"""
        x = self.input_norm(x)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick with MPS safety"""
        if self.training:
            # Clamp logvar to prevent numerical issues
            logvar = torch.clamp(logvar, -10, 10)
            std = torch.exp(0.5 * logvar)
            # Use device-safe random sampling
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def VAELoss(recon_x, x, mu, logvar, beta=1.0, alpha=0.05):
    """VAE loss function with KL divergence and L1 regularization"""
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    logvar = torch.clamp(logvar, -10, 10)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    l1_loss = alpha * torch.mean(torch.abs(mu))
    total_loss = recon_loss + beta * kl_loss + l1_loss

    # NaN fallback - return large finite value if NaN/Inf detected
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("Warning: NaN or Inf detected in loss, returning fallback loss")
        total_loss = torch.tensor(1e6, device=total_loss.device, requires_grad=True)
        recon_loss = torch.tensor(1e6, device=recon_loss.device)
        kl_loss = torch.tensor(0.0, device=kl_loss.device)
        l1_loss = torch.tensor(0.0, device=l1_loss.device)

    return total_loss, recon_loss, kl_loss, l1_loss
