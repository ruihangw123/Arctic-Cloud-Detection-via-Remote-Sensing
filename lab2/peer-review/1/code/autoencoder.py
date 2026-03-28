import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        optimizer_config=None,
        n_input_channels=8,
        patch_size=9,
        embedding_size=32,
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config
        self.n_input_channels = n_input_channels
        self.patch_size = patch_size
        self.embedding_size = embedding_size

        # This architecture assumes the default 9x9 patch size used in the lab.
        # After two stride-2 convolutions:
        # 9x9 -> 5x5 -> 3x3
        if patch_size != 9:
            raise ValueError(
                f"This convolutional autoencoder is currently configured for patch_size=9, "
                f"but got patch_size={patch_size}."
            )

        # Encoder:
        # Preserve local spatial structure instead of flattening immediately.
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=n_input_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # (B, 8, 9, 9) -> (B, 16, 9, 9)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # (B, 16, 9, 9) -> (B, 32, 5, 5)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # (B, 32, 5, 5) -> (B, 64, 3, 3)
            nn.ReLU(),
        )

        # Fully connected bottleneck.
        self.encoder_fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size),
        )

        # Decoder:
        # First map embedding back to a small feature map,
        # then use transpose convolutions to reconstruct the patch.
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 3 * 3),
            nn.ReLU(),
        )

        self.decoder_cnn = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(64, 3, 3)),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),  # (B, 64, 3, 3) -> (B, 32, 5, 5)
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),  # (B, 32, 5, 5) -> (B, 16, 9, 9)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=n_input_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # (B, 16, 9, 9) -> (B, 8, 9, 9)
        )

    def encode(self, x):
        """
        Encodes the input patch into a low-dimensional embedding.

        Args:
            x: Tensor of shape (batch_size, n_input_channels, patch_size, patch_size)

        Returns:
            Tensor of shape (batch_size, embedding_size)
        """
        x = self.encoder_cnn(x)
        x = self.encoder_fc(x)
        return x

    def decode(self, z):
        """
        Decodes the latent embedding back into a reconstructed patch.

        Args:
            z: Tensor of shape (batch_size, embedding_size)

        Returns:
            Tensor of shape (batch_size, n_input_channels, patch_size, patch_size)
        """
        z = self.decoder_fc(z)
        z = self.decoder_cnn(z)
        return z

    def forward(self, batch):
        """
        Forward pass through the autoencoder.

        Args:
            batch: Tensor of shape (batch_size, n_input_channels, patch_size, patch_size)

        Returns:
            Reconstructed tensor of the same shape as the input.
        """
        encoded = self.encode(batch)
        decoded = self.decode(encoded)
        return decoded

    def _compute_loss(self, batch):
        """
        Computes reconstruction loss and optional regularization terms.

        Args:
            batch: Tensor of shape (batch_size, n_input_channels, patch_size, patch_size)

        Returns:
            total_loss: Scalar tensor used for optimization
            recon_loss: Pure reconstruction MSE
            sparse_penalty: L1 penalty on the embedding
        """
        encoded = self.encode(batch)
        decoded = self.decode(encoded)

        # Reconstruction loss
        recon_loss = F.mse_loss(decoded, batch)

        # Small L1 penalty to encourage a more compact / sparse embedding
        sparse_lambda = self.optimizer_config.get("sparse_lambda", 0.0)
        sparse_penalty = sparse_lambda * encoded.abs().mean()

        total_loss = recon_loss + sparse_penalty
        return total_loss, recon_loss, sparse_penalty

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Logs epoch-level metrics so the loss history is meaningful.
        """
        loss, recon_loss, sparse_penalty = self._compute_loss(batch)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train_sparse_penalty", sparse_penalty, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Logs epoch-level metrics for consistent monitoring.
        """
        loss, recon_loss, sparse_penalty = self._compute_loss(batch)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val_sparse_penalty", sparse_penalty, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and optional learning-rate scheduler.

        Supported optimizer_config keys include:
            - lr
            - weight_decay
            - sparse_lambda
            - use_scheduler
            - scheduler_factor
            - scheduler_patience
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optimizer_config.get("lr", 1e-3),
            weight_decay=self.optimizer_config.get("weight_decay", 0.0),
        )

        use_scheduler = self.optimizer_config.get("use_scheduler", False)
        if not use_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.optimizer_config.get("scheduler_factor", 0.5),
            patience=self.optimizer_config.get("scheduler_patience", 3),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def embed(self, x):
        """
        Returns the latent embedding of the input tensor.

        Args:
            x: Tensor of shape (batch_size, n_input_channels, patch_size, patch_size)

        Returns:
            Tensor of shape (batch_size, embedding_size)
        """
        return self.encode(x)