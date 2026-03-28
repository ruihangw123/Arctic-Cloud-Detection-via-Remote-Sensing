"""
Inputs  : ../data/*.npz  (raw MISR labeled images)
           ../code/       (autoencoder.py, patchdataset.py, data.py)
Outputs : ../train_features.csv
           ../test_features.csv
           ../perturbed_dataset/test_features_perturbed.csv

Run: python feature_engineering.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Paths 
HERE = Path(__file__).resolve().parent   # .../code/
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
CODE_DIR = HERE                              # same directory as this script
EXPORT_DIR = ROOT / "feature_eng_dataset"
PERTURBED_DIR = EXPORT_DIR / "perturbed_dataset"
CKPT_DIR = HERE / "checkpoints"

EXPORT_DIR.mkdir(exist_ok=True)
PERTURBED_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(CODE_DIR))


from data import make_data
from autoencoder import Autoencoder
from patchdataset import PatchDataset

# Constants 
LABELED = ["O013257", "O013490", "O012791"]
TEST_IMAGE = "O013490"
COL_NAMES = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "label"]
RADIANCE_COLS = ["DF", "CF", "BF", "AF", "AN"]
PATCH_SIZE = 9
EMBEDDING_SIZE = 32
AE_COLS = [f"ae{i}" for i in range(EMBEDDING_SIZE)]
NOISE_LEVEL = 0.05

FINAL_FEATURES = [
    "SD", "CORR",
    "DF", "CF", "BF", "AF", "AN",
    "NDAI_DF_AF",
    "PC1",
]
FINAL_FEATURES_AE = FINAL_FEATURES + AE_COLS


# Step 1: Load raw data
print("Step 1: Loading raw data")

dfs = []
for name in LABELED:
    npz = np.load(DATA_DIR / f"{name}.npz")
    key = list(npz.files)[0]
    arr = npz[key]
    df_img = pd.DataFrame(arr, columns=COL_NAMES)
    df_img["image"] = name
    dfs.append(df_img)
    print(f"  Loaded {name}: {arr.shape[0]:,} pixels")

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total pixels: {len(df_all):,}\n")


# Step 2: Train / Test split
print("Step 2: Train/test split (temporal holdout)")

df_split = df_all[df_all["label"] != 0].copy()
df_train = df_split[df_split["image"] != TEST_IMAGE].copy()
df_test  = df_split[df_split["image"] == TEST_IMAGE].copy()

print(f"Train : {len(df_train):,} pixels  (O012791 + O013257)")
print(f"Test  : {len(df_test):,} pixels  ({TEST_IMAGE})\n")


# Step 3: Engineered features 
print("Step 3: Feature engineering")

# 3a. NDAI_DF_AF ratio variant
for df in [df_train, df_test]:
    df["NDAI_DF_AF"] = (df["DF"] - df["AF"]) / (df["DF"] + df["AF"] + 1e-8)
print("  NDAI_DF_AF created (replaces original NDAI)")

# 3b. PCA on 5 radiance angles, fit on train only!
scaler = StandardScaler()
X_rad_train = scaler.fit_transform(df_train[RADIANCE_COLS])
X_rad_test  = scaler.transform(df_test[RADIANCE_COLS])

pca = PCA(n_components=None, random_state=42)
pca.fit(X_rad_train)

pca_train = pca.transform(X_rad_train)
pca_test = pca.transform(X_rad_test)

df_train["PC1"] = pca_train[:, 0]
df_test["PC1"] = pca_test[:, 0]

var_explained = pca.explained_variance_ratio_[:2] * 100
print(f"PC1 created  (explains {var_explained[0]:.1f}% of radiance variance)")
print(f"PC2 explains {var_explained[1]:.1f}% \n")


# Step 4: Autoencoder 
print("Step 4: Autoencoder — spatial patch embeddings")

# 4a. Load patches
print("Loading patches (2000 per image x 164 images)...")
images_long, patches = make_data(patch_size=PATCH_SIZE, n_per_image=2000, seed=42)
all_patches = [p for image_patches in patches for p in image_patches]
print(f"Total patches: {len(all_patches):,}  shape: {all_patches[0].shape}")

# 4b. Normalization stats from all 164 images
all_raw = np.concatenate(
    [img[:, 2:10].astype(np.float32) for img in images_long], axis=0
)
ae_mean = all_raw.mean(axis=0)
ae_std  = all_raw.std(axis=0) + 1e-8
print(f"Normalization computed from {len(all_raw):,} pixels")

# 4c. Build DataLoaders
rng_split  = np.random.default_rng(42)
train_bool = rng_split.random(len(all_patches)) < 0.8
train_idx  = np.where( train_bool)[0]
val_idx    = np.where(~train_bool)[0]

train_loader = DataLoader(
    PatchDataset([all_patches[i] for i in train_idx]),
    batch_size=2048, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    PatchDataset([all_patches[i] for i in val_idx]),
    batch_size=2048, shuffle=False, num_workers=0
)

# 4d. Train autoencoder
model = Autoencoder(
    optimizer_config={"lr": 1e-3},
    n_input_channels=8,
    patch_size=PATCH_SIZE,
    embedding_size=EMBEDDING_SIZE,
)


trainer = L.Trainer(
    max_epochs=50,
    accelerator="auto",
    devices=1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True),
        ModelCheckpoint(
            monitor="val_loss", save_top_k=1,
            dirpath=str(CKPT_DIR), filename="ae-best"
        ),
    ],
    enable_progress_bar=True,
    log_every_n_steps=20,
    logger=False,
)

print("Training autoencoder...")
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print("Training complete.\n")


# Step 5: Extract embeddings
print("Step 5: Extracting embeddings for labeled images")

def get_embeddings(name, model, ae_mean, ae_std):
    pad = PATCH_SIZE // 2
    npz = np.load(DATA_DIR / f"{name}.npz")
    arr = npz[list(npz.files)[0]].astype(np.float32)

    y_coords = arr[:, 0].astype(int)
    x_coords = arr[:, 1].astype(int)
    features = arr[:, 2:10]

    y0, x0 = y_coords.min(), x_coords.min()
    H = y_coords.max() - y0 + 1
    W = x_coords.max() - x0 + 1

    grid   = np.zeros((8, H, W), dtype=np.float32)
    grid[:, y_coords - y0, x_coords - x0] = features.T
    padded = np.pad(grid, ((0,0),(pad,pad),(pad,pad)), mode="reflect")

    n_pix   = len(arr)
    patches = np.stack([
        padded[:,
               (y_coords[i]-y0+pad)-pad : (y_coords[i]-y0+pad)+pad+1,
               (x_coords[i]-x0+pad)-pad : (x_coords[i]-x0+pad)+pad+1]
        for i in range(n_pix)
    ])
    patches = (patches - ae_mean[None,:,None,None]) / ae_std[None,:,None,None]

    model.eval()
    device  = next(model.parameters()).device
    all_emb = []
    with torch.no_grad():
        for start in range(0, n_pix, 4096):
            batch = torch.tensor(
                patches[start:start+4096], dtype=torch.float32
            ).to(device)
            all_emb.append(model.embed(batch).cpu().numpy())

    emb    = np.vstack(all_emb)
    df_emb = pd.DataFrame(emb, columns=AE_COLS)
    df_emb.insert(0, "y", y_coords)
    df_emb.insert(1, "x", x_coords)
    print(f"  {name}: {n_pix:,} pixels → embeddings {emb.shape}")
    return df_emb


# Save per-image CSVs and merge into df_train / df_test
df_train.drop(columns=AE_COLS, errors="ignore", inplace=True)
df_test.drop(columns=AE_COLS, errors="ignore", inplace=True)

for name in LABELED:
    df_emb   = get_embeddings(name, model, ae_mean, ae_std)
    out_path = EXPORT_DIR / f"{name}_ae.csv"
    df_emb.to_csv(out_path, index=False)

    for df_split in [df_train, df_test]:
        mask = df_split["image"] == name
        if mask.any():
            merged = df_split.loc[mask, ["y","x"]].merge(df_emb, on=["y","x"], how="left")
            for col in AE_COLS:
                df_split.loc[mask, col] = merged[col].values

nan_train = df_train[AE_COLS].isna().sum().sum()
nan_test  = df_test[AE_COLS].isna().sum().sum()
assert nan_train == 0 and nan_test == 0, "NaN found in AE embeddings"
print(f"Embeddings merged. NaN check passed.\n")


# Step 6: Save outputs
print("Step 6: Saving outputs")

# Train
df_train_out = df_train[["image","y","x","label"] + FINAL_FEATURES_AE].copy()
df_train_out.to_csv(EXPORT_DIR / "train_features.csv", index=False)
print(f"train_features.csv : {df_train_out.shape}")

# Test
df_test_out = df_test[["image","y","x","label"] + FINAL_FEATURES_AE].copy()
df_test_out.to_csv(EXPORT_DIR / "test_features.csv", index=False)
print(f"test_features.csv  : {df_test_out.shape}")

# Perturbed test (5% noise)
np.random.seed(42)
noise = np.random.normal(0, NOISE_LEVEL, size=df_test[FINAL_FEATURES_AE].shape)
stds  = df_test[FINAL_FEATURES_AE].std().values

df_test_perturbed = df_test.copy()
df_test_perturbed[FINAL_FEATURES_AE] += noise * stds

df_test_perturbed_out = df_test_perturbed[["image","y","x","label"] + FINAL_FEATURES_AE].copy()
df_test_perturbed_out.to_csv(PERTURBED_DIR / "test_features_perturbed.csv", index=False)
print(f"test_features_perturbed.csv : {df_test_perturbed_out.shape}")

print("\nDone. All outputs saved")
