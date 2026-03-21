import lightning as L
import torch
import torch.nn as nn


class ConvAutoencoder(L.LightningModule):
    """Conv AE for 9x9 multi-channel patches with denoising + channel masking."""

    def __init__(
        self,
        optimizer_config=None,
        n_input_channels=8,
        patch_size=9,
        embedding_size=16,
        noise_std=0.1,
        channel_mask_prob=0.15,
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config
        self.n_input_channels = n_input_channels
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.noise_std = noise_std
        self.channel_mask_prob = channel_mask_prob

        # Encoder: 9x9 -> 7x7 -> 5x5 -> flatten -> embedding
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # flatten 64*5*5 -> embedding
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size),
        )

        # Decoder: mirror of encoder using ConvTranspose2d
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 5 * 5),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_input_channels, kernel_size=3, padding=1),
        )

    def encoder(self, x):
        h = self.encoder_conv(x)
        return self.encoder_fc(h)

    def decoder(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 5, 5)
        return self.decoder_conv(h)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def embed(self, x):
        return self.encoder(x)

    def _corrupt(self, x):
        """Channel masking + Gaussian noise for denoising."""
        # Channel masking: randomly zero out entire channels per sample
        mask = torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > self.channel_mask_prob
        x = x * mask
        # Additive Gaussian noise
        x = x + self.noise_std * torch.randn_like(x)
        return x

    def training_step(self, batch, batch_idx):
        corrupted = self._corrupt(batch)
        loss = nn.functional.mse_loss(self(corrupted), batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = nn.functional.mse_loss(batch, self(batch))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_config)


if __name__ == "__main__":
    model = ConvAutoencoder(embedding_size=16)
    x = torch.randn(4, 8, 9, 9)
    z = model.encoder(x)
    x_hat = model(x)
    print(f"Input:     {x.shape}")
    print(f"Embedding: {z.shape}")
    print(f"Recon:     {x_hat.shape}")
    assert x_hat.shape == x.shape, "Shape mismatch!"
    print("OK")
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")