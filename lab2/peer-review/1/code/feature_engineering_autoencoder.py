# EXAMPLE USAGE:
# python feature_engineering_autoencoder.py configs/default.yaml

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


COL_NAMES = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "label"]
RADIANCE_COLS = ["DF", "CF", "BF", "AF", "AN"]


def load_image_df(name, data_dir, labeled=True):
    npz = np.load(Path(data_dir) / f"{name}.npz")
    arr = npz[list(npz.files)[0]]

    if labeled and arr.shape[1] == 11:
        df = pd.DataFrame(arr, columns=COL_NAMES)
    elif (not labeled) and arr.shape[1] == 10:
        df = pd.DataFrame(arr, columns=COL_NAMES[:-1])
        df["label"] = 0
    elif arr.shape[1] == 11:
        df = pd.DataFrame(arr, columns=COL_NAMES)
    else:
        raise ValueError(f"{name}: unexpected array shape {arr.shape}")

    df["image"] = name
    return df


def add_engineered_features(df, scaler=None, pca=None, fit=False):
    df = df.copy()

    # Ratio features
    df["NDAI_DF_AF"] = (df["DF"] - df["AF"]) / (df["DF"] + df["AF"] + 1e-8)
    df["NDAI_CF_AN"] = (df["CF"] - df["AN"]) / (df["CF"] + df["AN"] + 1e-8)
    df["NDAI_BF_AN"] = (df["BF"] - df["AN"]) / (df["BF"] + df["AN"] + 1e-8)
    df["NDAI_AF_AN"] = (df["AF"] - df["AN"]) / (df["AF"] + df["AN"] + 1e-8)
    df["NDAI_CF_AF"] = (df["CF"] - df["AF"]) / (df["CF"] + df["AF"] + 1e-8)

    X_rad = df[RADIANCE_COLS].values

    if fit:
        scaler = StandardScaler()
        X_rad_scaled = scaler.fit_transform(X_rad)

        pca = PCA(n_components=None, random_state=42)
        pca.fit(X_rad_scaled)
    else:
        if scaler is None or pca is None:
            raise ValueError("When fit=False, scaler and pca must be provided.")
        X_rad_scaled = scaler.transform(X_rad)

    pca_scores = pca.transform(X_rad_scaled)
    df["PC1"] = pca_scores[:, 0]
    df["PC2"] = pca_scores[:, 1]

    return df, scaler, pca


def merge_embeddings(df, image_names, embed_dir):
    df = df.copy()

    example_file = Path(embed_dir) / f"{image_names[0]}_ae_opt.csv"
    emb_example = pd.read_csv(example_file)
    ae_cols = [c for c in emb_example.columns if c.startswith("ae")]

    df.drop(columns=ae_cols, errors="ignore", inplace=True)

    for name in image_names:
        mask = df["image"] == name
        if not mask.any():
            continue

        df_emb = pd.read_csv(Path(embed_dir) / f"{name}_ae_opt.csv")

        merged = df.loc[mask, ["y", "x"]].merge(
            df_emb,
            on=["y", "x"],
            how="left",
        )

        if len(merged) != mask.sum():
            raise ValueError(f"{name}: row count changed after merge.")

        missing_cells = merged[ae_cols].isnull().sum().sum()
        if missing_cells > 0:
            raise ValueError(f"{name}: missing AE values after merge = {missing_cells}")

        for col in ae_cols:
            df.loc[mask, col] = merged[col].values

    return df, ae_cols


def main():
    config_path = sys.argv[1]
    config = yaml.safe_load(open(config_path, "r"))

    data_dir = config["data"].get("data_dir", "../data")
    embed_dir = config["embedding"]["output_dir"]
    output_dir = Path(config["feature_dataset"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_images = config["feature_dataset"]["labeled_images"]
    test_image = config["feature_dataset"]["test_image"]

    # Labeled base tables
    dfs = [load_image_df(name, data_dir, labeled=True) for name in labeled_images]
    df_all = pd.concat(dfs, ignore_index=True)

    # train: two labeled images, excluding unlabeled rows
    df_train = df_all[
        (df_all["image"] != test_image) & (df_all["label"] != 0)
    ].copy()

    # test: keep entire designated image, including label==0 rows if present
    df_test = df_all[
        df_all["image"] == test_image
    ].copy()

    # Feature engineering
    df_train, scaler, pca = add_engineered_features(df_train, fit=True)
    df_test, _, _ = add_engineered_features(df_test, scaler=scaler, pca=pca, fit=False)

    # Final handcrafted features
    FINAL_FEATURES = [
        "SD",
        "CORR",
        "DF", "CF", "BF", "AF", "AN",
        "NDAI_DF_AF",
        "PC1",
    ]

    # Merge AE embeddings
    df_train, ae_cols = merge_embeddings(df_train, labeled_images, embed_dir)
    df_test, _ = merge_embeddings(df_test, labeled_images, embed_dir)

    FINAL_FEATURES_AE = FINAL_FEATURES + ae_cols

    df_train_out = df_train[["image", "y", "x", "label"] + FINAL_FEATURES_AE].copy()
    df_test_out = df_test[["image", "y", "x", "label"] + FINAL_FEATURES_AE].copy()

    train_out_path = output_dir / "train_features_opt.csv"
    test_out_path = output_dir / "test_features_opt.csv"

    df_train_out.to_csv(train_out_path, index=False)
    df_test_out.to_csv(test_out_path, index=False)

    print(f"Saved: {train_out_path}")
    print(f"Saved: {test_out_path}")
    print(f"Train shape: {df_train_out.shape}")
    print(f"Test shape : {df_test_out.shape}")

    # Save combined labeled embeddings
    emb_dfs = []
    for name in labeled_images:
        df_emb = pd.read_csv(Path(embed_dir) / f"{name}_ae_opt.csv")
        df_emb["image"] = name
        emb_dfs.append(df_emb)

    df_embeddings = pd.concat(emb_dfs, ignore_index=True)
    combined_emb_path = output_dir / "embeddings_ae_opt.csv"
    df_embeddings.to_csv(combined_emb_path, index=False)
    print(f"Saved: {combined_emb_path}")

    # Quick null checks
    print("\nNull checks:")
    print("train null cells:", int(df_train_out.isnull().sum().sum()))
    print("test null cells :", int(df_test_out.isnull().sum().sum()))


if __name__ == "__main__":
    main()