import lightning as L
import torch
import torch.nn as nn


class Autoencoder(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9,
        embedding_size=8, use_conv=False, sparsity_weight=0.0
    ):
        super().__init__()
        self.save_hyperparameters()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config
        self.sparsity_weight = sparsity_weight

        input_size = int(n_input_channels * (patch_size ** 2))

        if use_conv:
            # ---------- Convolutional encoder ----------
            # Takes advantage of spatial structure in patches
            self.encoder = nn.Sequential(
                # (batch, 8, 9, 9) -> (batch, 32, 9, 9)
                nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1),
                # (batch, 32, 9, 9) -> (batch, 64, 5, 5)  [floor((9-3)/2)+1=4... use padding]
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                # (batch, 64, 5, 5) -> (batch, 128, 3, 3)
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                # Flatten: (batch, 128, 3, 3) -> (batch, 1152)
                nn.Flatten(),
                nn.Linear(128 * 3 * 3, embedding_size),
            )

            # ---------- Convolutional decoder ----------
            self.decoder = nn.Sequential(
                nn.Linear(embedding_size, 128 * 3 * 3),
                nn.LeakyReLU(0.1),
                nn.Unflatten(1, (128, 3, 3)),
                # (batch, 128, 3, 3) -> (batch, 64, 5, 5)
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                # (batch, 64, 5, 5) -> (batch, 32, 9, 9)
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1),
                # (batch, 32, 9, 9) -> (batch, 8, 9, 9)
                nn.Conv2d(32, n_input_channels, kernel_size=3, padding=1),
            )
        else:
            # ---------- Original MLP architecture (improved) ----------
            self.encoder = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(input_size, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, embedding_size),
            )

            self.decoder = nn.Sequential(
                nn.Linear(embedding_size, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, input_size),
                nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),
            )

    def forward(self, batch):
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def _compute_loss(self, batch):
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # Reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(batch, decoded)

        # Optional: sparsity regularization on the latent space
        # Encourages the embedding to be sparse (many near-zero values)
        if self.sparsity_weight > 0:
            sparsity_loss = torch.mean(torch.abs(encoded))
            total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        else:
            total_loss = recon_loss

        return total_loss, recon_loss

    def training_step(self, batch, batch_idx):
        total_loss, recon_loss = self._compute_loss(batch)
        self.log("train_loss", total_loss)
        self.log("train_recon_loss", recon_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss = self._compute_loss(batch)
        self.log("val_loss", total_loss)
        self.log("val_recon_loss", recon_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def embed(self, x):
        return self.encoder(x)
