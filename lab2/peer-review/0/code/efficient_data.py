import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class EfficientPatchDataset(Dataset):
    """
    Memory-efficient dataset that stores full images and extracts
    patches on-the-fly in __getitem__, instead of pre-computing
    all ~19M patches (which causes OOM).

    Each image is stored as a padded (n_channels, H, W) tensor.
    __getitem__ returns a (n_channels, patch_size, patch_size) patch
    by slicing into the stored image.
    """

    def __init__(self, image_arrays, coords_list, patch_size=9,
                 global_means=None, global_stds=None):
        """
        Args:
            image_arrays: list of (n_channels, H, W) numpy arrays (already gridded)
            coords_list: list of (N_i, 2) arrays with (row, col) indices into
                         the grid for each image's valid pixels
            patch_size: size of patches to extract
            global_means: (n_channels,) array for normalization
            global_stds: (n_channels,) array for normalization
        """
        self.patch_size = patch_size
        self.pad = patch_size // 2

        # Normalize images
        if global_means is not None and global_stds is not None:
            self.images = []
            for img in image_arrays:
                img_norm = (img - global_means[:, None, None]) / global_stds[:, None, None]
                # Pad with reflection
                img_padded = np.pad(
                    img_norm,
                    ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                    mode='reflect'
                )
                self.images.append(torch.tensor(img_padded, dtype=torch.float32))
            self.global_means = global_means
            self.global_stds = global_stds
        else:
            self.images = []
            for img in image_arrays:
                img_padded = np.pad(
                    img,
                    ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                    mode='reflect'
                )
                self.images.append(torch.tensor(img_padded, dtype=torch.float32))

        # Store coordinates for each image
        # coords_list[i] is shape (N_i, 2) with (row_idx, col_idx) into the grid
        self.coords_list = coords_list

        # Build index mapping: global_idx -> (image_idx, row, col)
        # This lets us map a flat index to the right image and pixel
        self.index_map = []
        for img_idx, coords in enumerate(coords_list):
            for row, col in coords:
                self.index_map.append((img_idx, int(row), int(col)))

        print(f"EfficientPatchDataset: {len(self.index_map)} patches "
              f"from {len(self.images)} images, patch_size={patch_size}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        img_idx, row, col = self.index_map[idx]
        # Account for padding offset
        r = row + self.pad
        c = col + self.pad
        patch = self.images[img_idx][
            :,
            r - self.pad: r + self.pad + 1,
            c - self.pad: c + self.pad + 1
        ]
        return patch


def load_and_prepare_images(patch_size=9, max_images=None):
    """
    Load .npz files and prepare them for EfficientPatchDataset.

    Returns:
        image_arrays: list of (n_channels, H, W) numpy arrays
        coords_list: list of (N_i, 2) arrays with grid coordinates
        global_means, global_stds: for normalization
        filepaths: list of file paths loaded
    """
    # Find data files
    filepaths = sorted(glob.glob("../data/image_data/*.npz"))
    if len(filepaths) == 0:
        filepaths = sorted(glob.glob("../data/*.npz"))
    if len(filepaths) == 0:
        raise FileNotFoundError("No .npz files found")

    if max_images is not None and max_images < len(filepaths):
        print(f"Limiting to {max_images} of {len(filepaths)} images")
        filepaths = filepaths[:max_images]

    print(f"Loading {len(filepaths)} images...")

    # Load raw data
    images_long = []
    for fp in filepaths:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        if data.shape[1] == 11:
            data = data[:, :-1]  # remove labels
        images_long.append(data)

    # Compute global grid bounds
    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)
    nchannels = images_long[0].shape[1] - 2

    print(f"Grid size: {height} x {width}, channels: {nchannels}")

    # Reshape each image onto the common grid
    image_arrays = []
    coords_list = []
    for img in images_long:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        y_rel = y - global_miny
        x_rel = x - global_minx

        image = np.zeros((nchannels, height, width), dtype=np.float32)
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]

        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]

        image_arrays.append(image)
        # Store the valid pixel coordinates (row, col) in the grid
        coords_list.append(np.column_stack([y_valid, x_valid]))

    print("Done reshaping images")

    # Compute global normalization stats
    all_images = np.stack(image_arrays, axis=0)  # (N, C, H, W)
    global_means = np.mean(all_images, axis=(0, 2, 3))
    global_stds = np.std(all_images, axis=(0, 2, 3))
    del all_images  # free memory

    print(f"Global means: {global_means}")
    print(f"Global stds:  {global_stds}")

    return image_arrays, coords_list, global_means, global_stds, filepaths
