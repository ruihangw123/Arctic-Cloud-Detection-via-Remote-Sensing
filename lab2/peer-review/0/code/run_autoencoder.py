# USAGE:
# Local test:  python run_autoencoder.py configs/local_test.yaml
# Full run:    python run_autoencoder.py configs/bridges2.yaml

import numpy as np
import sys
import os
import yaml
import gc
import torch
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from autoencoder import Autoencoder
from efficient_data import EfficientPatchDataset, load_and_prepare_images


def main():
    print("Loading config file")
    config_path = sys.argv[1]
    assert os.path.exists(config_path), f"Config file {config_path} not found"
    config = yaml.safe_load(open(config_path, "r"))

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading and preparing image data")
    max_images = config["data"].get("max_images", None)
    image_arrays, coords_list, global_means, global_stds, filepaths = \
        load_and_prepare_images(
            patch_size=config["data"]["patch_size"],
            max_images=max_images
        )

    # Split by IMAGE (not by pixel) for better generalization
    n_images = len(image_arrays)
    n_train = int(0.8 * n_images)

    # Shuffle image indices with a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    image_indices = rng.permutation(n_images)
    train_indices = image_indices[:n_train]
    val_indices = image_indices[n_train:]

    print(f"Train images: {len(train_indices)}, Val images: {len(val_indices)}")

    # Create train and val datasets
    train_arrays = [image_arrays[i] for i in train_indices]
    train_coords = [coords_list[i] for i in train_indices]
    val_arrays = [image_arrays[i] for i in val_indices]
    val_coords = [coords_list[i] for i in val_indices]

    # Free the originals
    del image_arrays, coords_list
    gc.collect()

    train_dataset = EfficientPatchDataset(
        train_arrays, train_coords,
        patch_size=config["data"]["patch_size"],
        global_means=global_means, global_stds=global_stds
    )
    val_dataset = EfficientPatchDataset(
        val_arrays, val_coords,
        patch_size=config["data"]["patch_size"],
        global_means=global_means, global_stds=global_stds
    )

    # Free raw arrays after dataset creation
    del train_arrays, train_coords, val_arrays, val_coords
    gc.collect()

    # Create dataloaders
    dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
    dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

    print("Initializing model")
    ae_config = {k: v for k, v in config["autoencoder"].items()}
    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **ae_config,
    )
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("Preparing for training")
    checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

    # Trainer config
    trainer_config = {k: v for k, v in config["trainer"].items()}

    # Try wandb, fall back gracefully
    logger = None
    if "wandb" in config:
        try:
            from lightning.pytorch.loggers import WandbLogger
            if "SLURM_JOB_ID" in os.environ:
                config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
            logger = WandbLogger(config=config, **config["wandb"])
        except Exception as e:
            print(f"WandB not available, training without logging: {e}")

    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        **trainer_config,
    )

    print("Training")
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training complete!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")

    # Save normalization stats for later use in get_embedding.py
    np.savez("normalization_stats.npz",
             global_means=global_means, global_stds=global_stds)
    print("Saved normalization stats to normalization_stats.npz")


if __name__ == "__main__":
    main()
