# EXAMPLE USAGE:
# python get_embedding.py configs/default.yaml ../results/checkpoints/ae_opt-best.ckpt

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from autoencoder import Autoencoder


def load_stats(stats_path):
    stats_npz = np.load(stats_path)
    return {
        "mean": stats_npz["mean"],
        "std": stats_npz["std"],
        "global_miny": int(stats_npz["global_miny"]),
        "global_minx": int(stats_npz["global_minx"]),
        "height": int(stats_npz["height"]),
        "width": int(stats_npz["width"]),
        "patch_size": int(stats_npz["patch_size"]),
        "nchannels": int(stats_npz["nchannels"]),
    }


def get_embeddings_for_image(
    image_name,
    model,
    stats,
    data_dir,
    patch_size=9,
    batch_size=2048,
    device="cpu",
):
    """
    Build normalized 9x9 patches for one image using the exact same
    global-grid convention as training, then extract AE embeddings.
    """
    npz = np.load(Path(data_dir) / f"{image_name}.npz")
    arr = npz[list(npz.files)[0]].astype(np.float32)

    y_coords = arr[:, 0].astype(int)
    x_coords = arr[:, 1].astype(int)
    features = arr[:, 2:10]  # NDAI, SD, CORR, DF, CF, BF, AF, AN

    mean = stats["mean"]
    std = stats["std"]
    global_miny = stats["global_miny"]
    global_minx = stats["global_minx"]
    height = stats["height"]
    width = stats["width"]

    pad = patch_size // 2

    # Build global image grid
    grid = np.zeros((features.shape[1], height, width), dtype=np.float32)
    y_rel = y_coords - global_miny
    x_rel = x_coords - global_minx
    grid[:, y_rel, x_rel] = features.T

    # Same normalization as training
    grid = (grid - mean[:, None, None]) / std[:, None, None]

    # Same reflect padding as training
    padded = np.pad(grid, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")

    patches = np.stack([
        padded[
            :,
            (y_coords[i] - global_miny + pad) - pad:(y_coords[i] - global_miny + pad) + pad + 1,
            (x_coords[i] - global_minx + pad) - pad:(x_coords[i] - global_minx + pad) + pad + 1,
        ]
        for i in range(len(arr))
    ]).astype(np.float32)

    all_emb = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            batch = torch.tensor(
                patches[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            emb = model.embed(batch).cpu().numpy()
            all_emb.append(emb)

    emb = np.vstack(all_emb)
    ae_cols = [f"ae{i}" for i in range(emb.shape[1])]

    df_emb = pd.DataFrame(emb, columns=ae_cols)
    df_emb.insert(0, "y", y_coords)
    df_emb.insert(1, "x", x_coords)

    return df_emb


def main():
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    config = yaml.safe_load(open(config_path, "r"))

    stats = load_stats(config["output"]["stats_path"])

    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )

    map_location = None if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    data_dir = config["data"].get("data_dir", "../data")
    output_dir = Path(config["embedding"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_images = config["feature_dataset"]["labeled_images"]
    filepaths = [
        Path(data_dir) / f"{name}.npz"
        for name in labeled_images
    ]
    
    batch_size = config["embedding"].get("batch_size", 2048)
    
    print(f"Extracting embeddings for {len(filepaths)} labeled images...")

    for fp in tqdm(filepaths):
        image_name = fp.stem

        df_emb = get_embeddings_for_image(
            image_name=image_name,
            model=model,
            stats=stats,
            data_dir=data_dir,
            patch_size=config["data"]["patch_size"],
            batch_size=batch_size,
            device=device,
        )

        out_path = output_dir / f"{image_name}_ae_opt.csv"
        df_emb.to_csv(out_path, index=False)

    print("All embeddings exported.")


if __name__ == "__main__":
    main()