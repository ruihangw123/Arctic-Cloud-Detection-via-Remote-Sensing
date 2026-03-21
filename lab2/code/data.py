import glob
import numpy as np


FEATURE_COLS = ["NDAI", "SD", "CORR", "Rad_DF", "Rad_CF", "Rad_BF", "Rad_AF", "Rad_AN"]


def _load_npz_array(fp, keep_label=False):
    npz_data = np.load(fp)
    key = list(npz_data.files)[0]
    arr = npz_data[key]
    if (not keep_label) and arr.shape[1] == 11:
        arr = arr[:, :-1]
    return arr


def _rows_to_image_grid(data):
    """Convert row-format data into (channels, H, W) grid."""
    y = data[:, 0].astype(int)
    x = data[:, 1].astype(int)
    miny, maxy = y.min(), y.max()
    minx, maxx = x.min(), x.max()
    height = int(maxy - miny + 1)
    width = int(maxx - minx + 1)
    nchannels = data.shape[1] - 2

    image = np.zeros((nchannels, height, width), dtype=np.float32)
    y_rel = y - miny
    x_rel = x - minx
    for c in range(nchannels):
        image[c, y_rel, x_rel] = data[:, c + 2]
    return image, y_rel, x_rel


def _normalize_image_channels(image):
    """Channel-wise standardization for one image."""
    for c in range(image.shape[0]):
        mean = image[c].mean()
        std = image[c].std()
        if std > 0:
            image[c] = (image[c] - mean) / std
    return image


def make_data_low_ram(
    patch_size=9,
    n_images=None,
    max_patches=None,
    random_state=42,
    normalize_per_image=True,
):
    """Low-memory version of make_data -- processes images independently."""
    rng = np.random.default_rng(random_state)
    filepaths = sorted(glob.glob("../data/*.npz"))
    if n_images is not None:
        filepaths = filepaths[:n_images]

    pad_len = patch_size // 2
    patches = []

    for fp in filepaths:
        data = _load_npz_array(fp, keep_label=False)
        image, y_rel, x_rel = _rows_to_image_grid(data)

        if normalize_per_image:
            image = _normalize_image_channels(image)

        image_padded = np.pad(
            image,
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )

        for yy, xx in zip(y_rel, x_rel):
            patch = image_padded[:, yy : yy + patch_size, xx : xx + patch_size]
            patches.append(patch.astype(np.float32))

    if max_patches is not None and len(patches) > max_patches:
        idx = rng.choice(len(patches), size=max_patches, replace=False)
        patches = [patches[i] for i in idx]

    return filepaths, patches


def make_labeled_table_low_ram(
    labeled_files=("O012791.npz", "O013257.npz", "O013490.npz"),
    winsorize=False,
    winsor_quantiles=(0.01, 0.99),
):
    """Build labeled feature table (X, y, meta) from the three labeled images."""
    rows = []
    labels = []

    for fname in labeled_files:
        fp = f"../data/{fname}"
        arr = _load_npz_array(fp, keep_label=True)
        if arr.shape[1] != 11:
            continue

        feat = arr[:, 2:10].astype(np.float32)
        lbl = arr[:, 10].astype(int)

        mask = lbl != 0
        feat = feat[mask]
        lbl = (lbl[mask] == 1).astype(np.int64)

        rows.append(feat)
        labels.append(lbl)

    X = np.concatenate(rows, axis=0)
    y = np.concatenate(labels, axis=0)

    if winsorize:
        q_low, q_high = winsor_quantiles
        low = np.quantile(X, q_low, axis=0)
        high = np.quantile(X, q_high, axis=0)
        X = np.clip(X, low, high)

    meta = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "feature_names": FEATURE_COLS,
    }
    return X, y, meta


def make_data(patch_size=9):
    """Load all images and extract patches on a global grid."""
    filepaths = sorted(glob.glob("../data/*.npz"))
    images_long = []
    for fp in filepaths:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        if data.shape[1] == 11:
            data = data[:, :-1]
        images_long.append(data)

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
        image = np.zeros((nchannels, height, width))
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        images.append(image)
    print('done reshaping images')

    images = np.array(images)
    pad_len = patch_size // 2

    means = np.mean(images, axis=(0, 2, 3))[:, None, None]
    stds = np.std(images, axis=(0, 2, 3))[:, None, None]
    images = (images - means) / stds

    patches = []
    for i in range(len(images_long)):
        if i % 10 == 0:
            print(f'working on image {i}')
        patches_img = []
        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )
        ys = images_long[i][:, 0].astype(int)
        xs = images_long[i][:, 1].astype(int)
        for y, x in zip(ys, xs):
            y_idx = int(y - global_miny + pad_len)
            x_idx = int(x - global_minx + pad_len)
            patch = img_mirror[
                :,
                y_idx - pad_len : y_idx + pad_len + 1,
                x_idx - pad_len : x_idx + pad_len + 1,
            ]
            patches_img.append(patch.astype(np.float32))
        patches.append(patches_img)

    return images_long, patches


def load_single_image_patches(fp, patch_size=9):
    data = np.load(fp)
    key = list(data.files)[0]
    arr = data[key]
    if arr.shape[1] == 11:
        arr = arr[:, :-1]
    
    y = arr[:, 0].astype(int)
    x = arr[:, 1].astype(int)
    miny, minx = y.min(), x.min()
    height = int(y.max() - miny + 1)
    width  = int(x.max() - minx + 1)
    nchannels = arr.shape[1] - 2
    
    grid = np.zeros((nchannels, height, width), dtype=np.float32)
    for c in range(nchannels):
        grid[c, y - miny, x - minx] = arr[:, c + 2]
    
    means = grid.mean(axis=(1,2), keepdims=True)
    stds  = grid.std(axis=(1,2), keepdims=True) + 1e-8
    grid  = (grid - means) / stds
    
    pad_len = patch_size // 2
    img_mirror = np.pad(grid,
                        ((0,0),(pad_len,pad_len),(pad_len,pad_len)),
                        mode="reflect")
    patches = []
    for yi, xi in zip(y - miny, x - minx):
        patch = img_mirror[:, yi:yi+patch_size, xi:xi+patch_size]
        patches.append(patch.astype(np.float32))
    
    return arr, patches