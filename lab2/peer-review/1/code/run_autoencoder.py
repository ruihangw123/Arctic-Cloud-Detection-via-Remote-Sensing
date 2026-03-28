# EXAMPLE USAGE:
# python run_autoencoder.py default.yaml

import os
import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import WandbLogger

from autoencoder import Autoencoder
from patchdataset import PatchDataset
from data import make_data


class LossHistoryCallback(Callback):
    """
    Save epoch-level train/val metrics for later plotting in the report.
    """

    def __init__(self):
        super().__init__()
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        row = {
            "epoch": trainer.current_epoch,
            "train_loss": None,
            "train_recon_loss": None,
            "train_sparse_penalty": None,
            "val_loss": None,
            "val_recon_loss": None,
            "val_sparse_penalty": None,
            "lr": None,
        }

        for key in row:
            if key == "epoch":
                continue
            if key in metrics:
                value = metrics[key]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().item()
                row[key] = value

        try:
            optimizer = trainer.optimizers[0]
            row["lr"] = optimizer.param_groups[0]["lr"]
        except Exception:
            row["lr"] = None

        self.history.append(row)

    def to_dataframe(self):
        return pd.DataFrame(self.history)


def main():
    print("Loading config file...")
    config_path = sys.argv[1]
    assert os.path.exists(config_path), f"Config file {config_path} not found"
    config = yaml.safe_load(open(config_path, "r"))

    gc.collect()
    torch.cuda.empty_cache()

    seed = config["data"].get("seed", 42)
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    print("Making patch data...")
    images_long, patches, stats, image_info = make_data(
        patch_size=config["data"]["patch_size"],
        n_per_image=config["data"].get("n_per_image", None),
        seed=seed,
        data_dir=config["data"].get("data_dir", "../data"),
        return_stats=True,
        return_image_info=True,
    )

    # Image-level split
    n_images = len(patches)
    image_ids = np.arange(n_images)
    rng = np.random.default_rng(config["split"].get("seed", 42))
    rng.shuffle(image_ids)

    split_idx = int(config["split"].get("train_ratio", 0.8) * n_images)
    train_image_ids = set(image_ids[:split_idx])
    val_image_ids = set(image_ids[split_idx:])

    train_patches = [
        patch
        for img_idx, image_patches in enumerate(patches)
        if img_idx in train_image_ids
        for patch in image_patches
    ]
    val_patches = [
        patch
        for img_idx, image_patches in enumerate(patches)
        if img_idx in val_image_ids
        for patch in image_patches
    ]

    print(f"Train images  : {len(train_image_ids)}")
    print(f"Val images    : {len(val_image_ids)}")
    print(f"Train patches : {len(train_patches):,}")
    print(f"Val patches   : {len(val_patches):,}")

    train_dataset = PatchDataset(train_patches)
    val_dataset = PatchDataset(val_patches)

    # If no GPU, avoid persistent_workers issue on some systems
    train_loader_cfg = dict(config["dataloader_train"])
    val_loader_cfg = dict(config["dataloader_val"])

    if not torch.cuda.is_available():
        train_loader_cfg["pin_memory"] = False
        val_loader_cfg["pin_memory"] = False

    if train_loader_cfg.get("num_workers", 0) == 0:
        train_loader_cfg["persistent_workers"] = False
    if val_loader_cfg.get("num_workers", 0) == 0:
        val_loader_cfg["persistent_workers"] = False

    dataloader_train = DataLoader(train_dataset, **train_loader_cfg)
    dataloader_val = DataLoader(val_dataset, **val_loader_cfg)

    print("Initializing model...")
    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )
    print(model)

    checkpoint_dir = Path(config["checkpoint"]["dirpath"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stats_path = Path(config["output"]["stats_path"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    loss_history_path = Path(config["output"]["loss_history_path"])
    loss_history_path.parent.mkdir(parents=True, exist_ok=True)

    pt_path = Path(config["output"]["pt_path"])
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=config["optimizer"].get("scheduler_patience", 4) + 6,
        mode="min",
        verbose=True,
    )

    loss_history_callback = LossHistoryCallback()

    if "SLURM_JOB_ID" in os.environ:
        config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

    # wandb_logger = WandbLogger(config=config, **config["wandb"])
    wandb_logger = None

    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            loss_history_callback,
        ],
        **config["trainer"],
    )

    print("Training...")
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    best_ckpt = checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_ckpt}")

    # Save stats for exact reuse
    print("Saving normalization stats...")
    np.savez(
        stats_path,
        mean=stats["mean"],
        std=stats["std"],
        global_miny=stats["global_miny"],
        global_minx=stats["global_minx"],
        height=stats["height"],
        width=stats["width"],
        patch_size=stats["patch_size"],
        nchannels=stats["nchannels"],
    )
    print(f"Saved stats to: {stats_path}")

    # Save loss history
    print("Saving loss history...")
    loss_history_df = loss_history_callback.to_dataframe()
    loss_history_df.to_csv(loss_history_path, index=False)
    print(f"Saved loss history to: {loss_history_path}")

    # Save .pt file required by lab
    print("Saving .pt model...")
    best_model = Autoencoder.load_from_checkpoint(
        best_ckpt,
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )
    torch.save(best_model.state_dict(), pt_path)
    print(f"Saved .pt to: {pt_path}")

    print("Done.")


if __name__ == "__main__":
    main()