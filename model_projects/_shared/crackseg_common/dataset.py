import json
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["background", "crack", "loss", "shrinkage", "craquelure"]
NUM_CLASSES = len(CLASS_NAMES)


def set_class_names(names):
    """執行期改 CLASS_NAMES / NUM_CLASSES (給 2/3-class 訓練用)。"""
    global CLASS_NAMES, NUM_CLASSES
    CLASS_NAMES = list(names)
    NUM_CLASSES = len(CLASS_NAMES)
    return CLASS_NAMES, NUM_CLASSES


def load_tile_index(tiles_root):
    path = os.path.join(tiles_root, "tile_index.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 tile_index.json: {path}")
    with open(path) as f:
        return json.load(f)


class TileSegDataset(Dataset):
    """讀取 (image_tile, mask_tile) 配對，輸出 normalized tensor 與 long mask。

    item: list of dict, 每個元素至少有 "tile" 欄位（檔名）。
    """

    def __init__(self, tiles_root, items, transforms=None):
        self.image_dir = os.path.join(tiles_root, "images")
        self.mask_dir = os.path.join(tiles_root, "masks")
        self.items = list(items)
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def _load(self, name):
        img = np.array(Image.open(os.path.join(self.image_dir, name)).convert("RGB"))
        msk = np.array(Image.open(os.path.join(self.mask_dir, name)))
        if msk.ndim == 3:
            msk = msk[..., 0]
        return img, msk.astype(np.uint8)

    def __getitem__(self, idx):
        item = self.items[idx]
        name = item["tile"] if isinstance(item, dict) else item
        img, msk = self._load(name)

        if self.transforms is not None:
            out = self.transforms(image=img, mask=msk)
            img_t = out["image"]
            msk_t = out["mask"]
            if isinstance(msk_t, torch.Tensor):
                msk_t = msk_t.long()
            else:
                msk_t = torch.from_numpy(np.asarray(msk_t)).long()
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            msk_t = torch.from_numpy(msk).long()

        return {
            "image": img_t,
            "mask": msk_t,
            "name": name,
        }


def compute_class_weights(items, tiles_root, num_classes=NUM_CLASSES, mode="median_freq"):
    """從 mask tile 統計類別像素數，回傳 per-class weight tensor。

    mode:
        "median_freq": w_c = median(freq) / freq_c
        "inv_sqrt":    w_c = sqrt(total / (num_classes * count_c))
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    mask_dir = os.path.join(tiles_root, "masks")
    for it in items:
        name = it["tile"] if isinstance(it, dict) else it
        msk = np.array(Image.open(os.path.join(mask_dir, name)))
        if msk.ndim == 3:
            msk = msk[..., 0]
        for c in range(num_classes):
            counts[c] += int((msk == c).sum())

    counts = counts.astype(np.float64)
    total = counts.sum()
    freq = counts / max(total, 1.0)

    if mode == "median_freq":
        valid = freq > 0
        med = np.median(freq[valid]) if valid.any() else 1.0
        w = np.where(valid, med / np.clip(freq, 1e-12, None), 0.0)
    elif mode == "inv_sqrt":
        w = np.sqrt(total / np.clip(num_classes * counts, 1e-12, None))
    else:
        raise ValueError(f"unknown mode: {mode}")

    return torch.tensor(w, dtype=torch.float32), counts.astype(np.int64)
