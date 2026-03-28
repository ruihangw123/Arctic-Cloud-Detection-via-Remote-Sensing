import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path

def verify_boundary_gap(file_path):
    """
    Computes the minimum Euclidean distance between pixels of opposite labels (+1 vs -1).
    """
    df = pd.read_csv(file_path)
    
    # Ensure necessary columns exist
    if not all(col in df.columns for col in ["x", "y", "label"]):
        print(f"Error: {file_path} must contain 'x', 'y', and 'label' columns.")
        return None

    # Split into two classes
    coords_cloud = df[df["label"] == 1][["x", "y"]].values
    coords_clear = df[df["label"] == -1][["x", "y"]].values

    if len(coords_cloud) == 0 or len(coords_clear) == 0:
        print(f"Warning: One or both classes are empty in {file_path}.")
        return None

    # Use KDTree for efficient nearest neighbor search
    print(f"Analyzing {Path(file_path).name} ({len(df)} pixels)...")
    tree_cloud = KDTree(coords_cloud)
    
    # Query distance to nearest cloud for every clear pixel
    dists, _ = tree_cloud.query(coords_clear, k=1)
    
    min_dist = np.min(dists)
    print(f"  - Empirical Minimum Distance to Boundary: {min_dist:.4f}")
    
    # Optional: check unique label transitions
    # If pixels are on a grid with spacing 1.0, a value > 1.0 means no adjacent opposite-label pairs exist.
    if min_dist > 1.01:
        print(f"  - CONFIRMED: There is a spatial labeling gap (> 1.0 unit).")
    else:
        print(f"  - NOTE: Opposite-label pixels are adjacent (dist <= 1.0).")
        
    return min_dist

if __name__ == "__main__":
    # Check the temporal split files
    DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "feature_eng_dataset"
    
    files_to_check = [
        DATASET_DIR / "train_features.csv",
        DATASET_DIR / "test_features.csv"
    ]
    
    for f in files_to_check:
        if f.exists():
            verify_boundary_gap(f)
        else:
            print(f"File not found: {f}")
