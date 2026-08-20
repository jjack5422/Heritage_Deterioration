import os
import numpy as np
from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(root):
    return sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.lower().endswith(IMG_EXTS)
    )


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 127).astype(np.uint8)


def tile_image(img, tile_size=1024, stride=512, pad_value=0):
    h, w = img.shape[:2]
    pad_h = max(0, tile_size - h) if h < tile_size else (stride - (h - tile_size) % stride) % stride
    pad_w = max(0, tile_size - w) if w < tile_size else (stride - (w - tile_size) % stride) % stride
    if pad_h or pad_w:
        if img.ndim == 3:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=pad_value)
        else:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=pad_value)
    H, W = img.shape[:2]
    tiles, coords = [], []
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tiles.append(img[y:y + tile_size, x:x + tile_size])
            coords.append((y, x))
    return tiles, coords, (H, W)


def stitch_tiles(tiles, coords, full_shape, tile_size=1024):
    H, W = full_shape
    canvas = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    for tile, (y, x) in zip(tiles, coords):
        canvas[y:y + tile_size, x:x + tile_size] += tile.astype(np.float32)
        weight[y:y + tile_size, x:x + tile_size] += 1.0
    weight[weight == 0] = 1.0
    return canvas / weight


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    out = image.copy()
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    m = mask.astype(bool)
    color = np.array(color, dtype=np.uint8)
    out[m] = (alpha * color + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def split_train_val(items, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * val_ratio)))
    return items[n_val:], items[:n_val]
