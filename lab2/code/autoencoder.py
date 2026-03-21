import argparse
import gc
import glob
import os

import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from conv_autoencoder import ConvAutoencoder
from data import make_data, make_data_low_ram, load_single_image_patches
from patchdataset import PatchDataset


def _get_model_class(config):
    model_type = config.get("model_type", "mlp")
    if model_type == "conv":
        return ConvAutoencoder
    return Autoencoder


class Autoencoder(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        input_size = int(n_input_channels * (patch_size**2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, embedding_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),
        )

    def forward(self, batch):
        """Encode then decode the input tensor."""
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """Compute MSE training loss."""
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        loss = torch.nn.functional.mse_loss(batch, decoded)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute MSE validation loss."""
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        loss = torch.nn.functional.mse_loss(batch, decoded)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """Run the encoder to get embeddings."""
        return self.encoder(x)


def _resolve_path(path_arg, fallback_path):
    if path_arg is None:
        return fallback_path
    if os.path.isabs(path_arg):
        return path_arg
    return os.path.normpath(os.path.join(os.getcwd(), path_arg))


def _load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_all_patches_full(config):
    _, patches = make_data(patch_size=config["data"]["patch_size"])
    all_patches = [patch for image_patches in patches for patch in image_patches]
    return all_patches


def _load_all_patches_local(config, n_images, max_patches, seed):
    _, all_patches = make_data_low_ram(
        patch_size=config["data"]["patch_size"],
        n_images=n_images,
        max_patches=max_patches,
        random_state=seed,
        normalize_per_image=True,
    )
    return all_patches


def _train_val_split(patches, seed=42):
    rng = np.random.default_rng(seed)
    train_bool = rng.random(len(patches)) < 0.8
    train_idx = np.where(train_bool)[0]
    val_idx = np.where(~train_bool)[0]
    train_patches = [patches[i] for i in train_idx]
    val_patches = [patches[i] for i in val_idx]
    return train_patches, val_patches


def _build_model(config):
    cls = _get_model_class(config)
    return cls(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )


def _run_training(config, all_patches, use_wandb=False):
    L.seed_everything(42, workers=True)
    train_patches, val_patches = _train_val_split(all_patches, seed=42)
    print(f"Train patches: {len(train_patches):,} | Val patches: {len(val_patches):,}")

    train_dataset = PatchDataset(train_patches)
    val_dataset = PatchDataset(val_patches)
    dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
    dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

    model = _build_model(config)
    checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

    logger = None
    if use_wandb:
        if "wandb" not in config:
            raise ValueError("Config missing `wandb` section for --use-wandb")
        from lightning.pytorch.loggers import WandbLogger

        if "SLURM_JOB_ID" in os.environ:
            config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
        logger = WandbLogger(config=config, **config["wandb"])
    else:
        logger = L.pytorch.loggers.CSVLogger("logs", name="autoencoder")

    trainer = L.Trainer(logger=logger, callbacks=[checkpoint_callback], **config["trainer"])
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    print("Training complete")
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score}")

    # Save best model as .pt to results/ (required by lab instructions)
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        results_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results"))
        os.makedirs(results_dir, exist_ok=True)
        pt_path = os.path.join(results_dir, "autoencoder_best.pt")
        best_model = _build_model(config)
        ckpt_data = torch.load(best_ckpt, map_location="cpu")
        best_model.load_state_dict(ckpt_data["state_dict"])
        torch.save(best_model.state_dict(), pt_path)
        print(f"Saved best model weights to: {pt_path}")


def _save_embeddings(config, checkpoint_path, output_dir):
    cls = _get_model_class(config)
    model = cls(
        patch_size=config["data"]["patch_size"],
        optimizer_config=config.get("optimizer", {}),
        **config["autoencoder"],
    )

    map_location = None if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()


    filepaths = sorted(glob.glob(os.path.join(output_dir, "..", "data", "*.npz")))
    if not filepaths:
        filepaths = sorted(glob.glob("../data/*.npz"))

    os.makedirs(output_dir, exist_ok=True)
    emb_dim = config["autoencoder"].get("embedding_size", 8)

    for fp in tqdm(filepaths, desc="Saving image embeddings"):
        arr, patches = load_single_image_patches(fp, patch_size=config["data"]["patch_size"])
        
        with torch.no_grad():
            emb = model.embed(torch.tensor(np.array(patches), dtype=torch.float32))
            emb = emb.detach().cpu().numpy()
        
        embedding_df = pd.DataFrame(emb, columns=[f"ae{j}" for j in range(emb_dim)])
        embedding_df["y"] = arr[:, 0]
        embedding_df["x"] = arr[:, 1]
        cols = embedding_df.columns.tolist()
        embedding_df = embedding_df[["y", "x"] + [c for c in cols if c not in ["y", "x"]]]
        stem = os.path.splitext(os.path.basename(fp))[0]
        embedding_df.to_csv(os.path.join(output_dir, f"{stem}_ae.csv"), index=False)
        
        del arr, patches, emb, embedding_df
        gc.collect()
    print(f"Embeddings saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified autoencoder script: model + train + embedding export.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_local",
        choices=["train_full", "train_local", "embed"],
        help="train_full: use make_data over full dataset; train_local: low-RAM training; embed: export embeddings from checkpoint",
    )
    parser.add_argument("--config", type=str, default="configs/local.yaml", help="Path to YAML config")
    parser.add_argument("--n-images", type=int, default=30, help="For train_local: number of .npz files to load")
    parser.add_argument("--max-patches", type=int, default=300000, help="For train_local: cap total patches (0 = no cap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Use WandB logger (default: on)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logger")
    parser.add_argument("--sweep", action="store_true", help="Run as wandb sweep agent (overrides hyperparams from wandb)")
    parser.add_argument("--sweep-config", type=str, default="configs/sweep.yaml", help="Sweep YAML config path")
    parser.add_argument("--sweep-count", type=int, default=12, help="Number of sweep runs")
    parser.add_argument("--checkpoint", type=str, default="", help="For embed mode: checkpoint path")
    parser.add_argument("--output-dir", type=str, default="../data", help="For embed mode: output CSV directory")
    return parser.parse_args()


def _apply_sweep_overrides(config):
    """Override config hyperparams from wandb sweep agent."""
    import wandb
    sweep_params = wandb.config
    if hasattr(sweep_params, "embedding_size"):
        config["autoencoder"]["embedding_size"] = sweep_params.embedding_size
    if hasattr(sweep_params, "lr"):
        config["optimizer"]["lr"] = sweep_params.lr
    if hasattr(sweep_params, "noise_std"):
        config["autoencoder"]["noise_std"] = sweep_params.noise_std
    if hasattr(sweep_params, "channel_mask_prob"):
        config["autoencoder"]["channel_mask_prob"] = sweep_params.channel_mask_prob
    # Update wandb run name to reflect params
    emb = config["autoencoder"]["embedding_size"]
    lr = config["optimizer"]["lr"]
    ns = config["autoencoder"].get("noise_std", 0)
    cm = config["autoencoder"].get("channel_mask_prob", 0)
    wandb.run.name = f"emb{emb}_lr{lr}_ns{ns}_cm{cm}"
    return config


def _sweep_train():
    """Single sweep run: init wandb, override config, train."""
    import wandb
    wandb.init()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", "autoencoder.yaml")
    config = _load_config(config_path)
    config = _apply_sweep_overrides(config)

    all_patches = _load_all_patches_local(config, n_images=30, max_patches=300000, seed=42)
    print(f"Sweep run | patches={len(all_patches):,} | "
          f"emb={config['autoencoder']['embedding_size']} lr={config['optimizer']['lr']}")
    _run_training(config, all_patches, use_wandb=True)
    wandb.finish()


def main():
    args = parse_args()

    if args.sweep:
        import wandb
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sweep_cfg_path = os.path.join(script_dir, args.sweep_config)
        sweep_id = wandb.sweep(
            sweep=_load_config(sweep_cfg_path),
            project="stat214-lab2",
        )
        wandb.agent(sweep_id, function=_sweep_train, count=args.sweep_count)
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = _resolve_path(args.config, os.path.join(script_dir, "configs", "local.yaml"))
    config = _load_config(config_path)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_wandb = args.use_wandb and not args.no_wandb

    if args.mode == "train_full":
        print("Mode: train_full")
        all_patches = _load_all_patches_full(config)
        print(f"Total patches loaded: {len(all_patches):,}")
        _run_training(config, all_patches, use_wandb=use_wandb)

    elif args.mode == "train_local":
        print("Mode: train_local")
        max_patches = args.max_patches if args.max_patches > 0 else None
        all_patches = _load_all_patches_local(
            config,
            n_images=args.n_images,
            max_patches=max_patches,
            seed=args.seed,
        )
        print(f"Total patches loaded: {len(all_patches):,}")
        _run_training(config, all_patches, use_wandb=use_wandb)

    else:
        print("Mode: embed")
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode embed")
        checkpoint_path = _resolve_path(args.checkpoint, args.checkpoint)
        output_dir = _resolve_path(args.output_dir, os.path.join(script_dir, "../data"))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        _save_embeddings(config, checkpoint_path=checkpoint_path, output_dir=output_dir)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
