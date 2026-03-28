import glob
from pathlib import Path
import numpy as np


def make_data(
    patch_size=9,
    n_per_image=None,
    seed=42,
    data_dir="../data",
    return_stats=False,
    return_image_info=False,
    dtype=np.float32,
):
    """
    Load MISR image data and create normalized spatial patches.

    Parameters
    ----------
    patch_size : int, default=9
        Size of the square patch to extract around each pixel.
        Must be an odd integer so there is a well-defined center pixel.

    n_per_image : int or None, default=None
        If provided, randomly sample this many pixel locations per image
        instead of extracting every pixel. This greatly reduces memory usage.

    seed : int, default=42
        Random seed used for reproducible sampling.

    data_dir : str, default="../data"
        Directory containing the .npz image files.

    return_stats : bool, default=False
        If True, also return the global channel-wise normalization mean/std
        used to normalize the image grids.

    return_image_info : bool, default=False
        If True, also return metadata for each image, including:
            - filepath
            - image_name
            - sampled_indices
            - original_num_pixels
            - sampled_num_pixels

    dtype : numpy dtype, default=np.float32
        Output dtype for image grids and patches.

    Returns
    -------
    images_long : list of np.ndarray
        Original per-image tabular arrays, with label column removed if present.
        Each array has columns:
            y, x, feature_1, ..., feature_8

    patches : list of list of np.ndarray
        patches[i] is a list of sampled patches for image i.
        Each patch has shape (nchannels, patch_size, patch_size).

    stats : dict, optional
        Returned only if return_stats=True.
        Contains:
            - mean: shape (nchannels,)
            - std:  shape (nchannels,)
            - global_miny
            - global_minx
            - height
            - width

    image_info : list of dict, optional
        Returned only if return_image_info=True.
        Metadata for each image.
    """
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd, but got {patch_size}.")

    if n_per_image is not None and n_per_image <= 0:
        raise ValueError(f"n_per_image must be positive or None, but got {n_per_image}.")

    filepaths = sorted(glob.glob(str(Path(data_dir) / "*.npz")))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No .npz files found in directory: {data_dir}")

    # -------------------------------------------------------------------------
    # Load all images in their original tabular format.
    # If a label column exists, remove it so the autoencoder remains unsupervised.
    # -------------------------------------------------------------------------
    images_long = []
    image_names = []

    for fp in filepaths:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]

        # Remove label column if present
        if data.shape[1] == 11:
            data = data[:, :-1]

        data = data.astype(dtype, copy=False)
        images_long.append(data)
        image_names.append(Path(fp).stem)

    # -------------------------------------------------------------------------
    # Build one common global grid across all images.
    # This matches the original implementation and ensures that all images
    # can be normalized consistently in a shared tensor shape.
    # -------------------------------------------------------------------------
    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)

    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()

    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)

    nchannels = images_long[0].shape[1] - 2

    images = []
    for img in images_long:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)

        y_rel = y - global_miny
        x_rel = x - global_minx

        image = np.zeros((nchannels, height, width), dtype=dtype)

        valid_mask = (
            (y_rel >= 0) & (y_rel < height) &
            (x_rel >= 0) & (x_rel < width)
        )

        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]

        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]

        images.append(image)

    print("Done reshaping images onto the common grid.")

    images = np.asarray(images, dtype=dtype)
    pad_len = patch_size // 2

    # -------------------------------------------------------------------------
    # Compute global channel-wise normalization statistics.
    # These are the statistics you should reuse later for embedding extraction.
    # -------------------------------------------------------------------------
    means = np.mean(images, axis=(0, 2, 3), dtype=np.float64).astype(dtype)
    stds = np.std(images, axis=(0, 2, 3), dtype=np.float64).astype(dtype)

    # Prevent division by zero in degenerate cases
    stds = np.where(stds == 0, 1.0, stds).astype(dtype)

    images = (images - means[None, :, None, None]) / stds[None, :, None, None]

    rng = np.random.default_rng(seed)

    patches = []
    image_info = []

    # -------------------------------------------------------------------------
    # Extract patches image by image.
    # If n_per_image is provided, sample pixel locations before extraction.
    # -------------------------------------------------------------------------
    for i in range(len(images_long)):
        if i % 10 == 0:
            print(f"Working on image {i + 1}/{len(images_long)}")

        img_table = images_long[i]
        ys_all = img_table[:, 0].astype(int)
        xs_all = img_table[:, 1].astype(int)

        original_num_pixels = len(ys_all)

        if n_per_image is not None and n_per_image < original_num_pixels:
            sampled_indices = rng.choice(
                original_num_pixels,
                size=n_per_image,
                replace=False,
            )
            sampled_indices = np.sort(sampled_indices)
        else:
            sampled_indices = np.arange(original_num_pixels)

        ys = ys_all[sampled_indices]
        xs = xs_all[sampled_indices]

        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )

        patches_img = []
        for y, x in zip(ys, xs):
            y_idx = int(y - global_miny + pad_len)
            x_idx = int(x - global_minx + pad_len)

            patch = img_mirror[
                :,
                y_idx - pad_len : y_idx + pad_len + 1,
                x_idx - pad_len : x_idx + pad_len + 1,
            ]

            patches_img.append(patch.astype(dtype, copy=False))

        patches.append(patches_img)

        if return_image_info:
            image_info.append({
                "filepath": filepaths[i],
                "image_name": image_names[i],
                "sampled_indices": sampled_indices,
                "original_num_pixels": original_num_pixels,
                "sampled_num_pixels": len(sampled_indices),
            })

    # -------------------------------------------------------------------------
    # Build flexible return values so old code still works by default.
    # -------------------------------------------------------------------------
    outputs = [images_long, patches]

    if return_stats:
        stats = {
            "mean": means,
            "std": stds,
            "global_miny": global_miny,
            "global_minx": global_minx,
            "height": height,
            "width": width,
            "patch_size": patch_size,
            "nchannels": nchannels,
        }
        outputs.append(stats)

    if return_image_info:
        outputs.append(image_info)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)