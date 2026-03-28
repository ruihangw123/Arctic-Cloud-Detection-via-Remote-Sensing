# Extract autoencoder embeddings for the 3 labeled images
# Run from lab2/code/ directory on Bridges-2:
#   python get_embedding.py configs/bridges2.yaml checkpoints/conv-ae-epoch=099.ckpt

import sys
import torch
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from autoencoder import Autoencoder
from efficient_data import load_and_prepare_images, EfficientPatchDataset


def main():
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    config = yaml.safe_load(open(config_path, "r"))
    patch_size = config["data"]["patch_size"]
    embedding_size = config["autoencoder"]["embedding_size"]

    print("Loading the saved model")
    model = Autoencoder(patch_size=patch_size, **config["autoencoder"])
    map_location = None if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load only the 3 labeled images
    labeled_files = ['O013257.npz', 'O013490.npz', 'O012791.npz']
    col_names = ['y', 'x', 'ndai', 'sd', 'corr',
                 'angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an',
                 'cloud']

    # Load normalization stats from training
    stats = np.load("normalization_stats.npz")
    global_means = stats["global_means"]
    global_stds = stats["global_stds"]
    print(f"Loaded normalization stats: means={global_means.shape}, stds={global_stds.shape}")

    import glob
    # Find data path
    data_dir = "../data/image_data"
    if not glob.glob(f"{data_dir}/*.npz"):
        data_dir = "../data"

    for fname in labeled_files:
        print(f"\nProcessing {fname}...")
        filepath = f"{data_dir}/{fname}"
        npz_data = np.load(filepath)
        key = list(npz_data.files)[0]
        raw_data = npz_data[key]

        # Build dataframe with original data
        df = pd.DataFrame(raw_data, columns=col_names)

        # Build the image grid for patch extraction
        y = raw_data[:, 0].astype(int)
        x = raw_data[:, 1].astype(int)
        miny, maxy = y.min(), y.max()
        minx, maxx = x.min(), x.max()
        height = int(maxy - miny + 1)
        width = int(maxx - minx + 1)
        nchannels = 8

        image = np.zeros((nchannels, height, width), dtype=np.float32)
        y_rel = y - miny
        x_rel = x - minx
        for c in range(nchannels):
            image[c, y_rel, x_rel] = raw_data[:, c + 2]

        # Create dataset for this image
        coords = np.column_stack([y_rel, x_rel])
        dataset = EfficientPatchDataset(
            [image], [coords],
            patch_size=patch_size,
            global_means=global_means,
            global_stds=global_stds
        )

        # Extract embeddings in batches
        batch_size = 8192
        all_embeddings = []

        with torch.no_grad():
            for start in tqdm(range(0, len(dataset), batch_size)):
                end = min(start + batch_size, len(dataset))
                batch = torch.stack([dataset[i] for i in range(start, end)])
                batch = batch.to(device)
                emb = model.embed(batch)
                all_embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"  Embeddings shape: {embeddings.shape}")

        # Create embedding columns
        emb_cols = [f"ae{i}" for i in range(embedding_size)]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols)

        # Combine with original data
        result = pd.concat([df, emb_df], axis=1)

        # Save
        out_name = fname.replace('.npz', '')
        out_path = f"../data/image_data/{out_name}_with_embeddings.csv"
        result.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print("\nDone! Embedding CSVs saved for all 3 labeled images.")


if __name__ == "__main__":
    main()
