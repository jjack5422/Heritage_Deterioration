import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(image_size=512, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, fill=0, fill_mask=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.CLAHE(clip_limit=(1, 2), tile_grid_size=(8, 8), p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def val_transforms(image_size=512, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, fill=0, fill_mask=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
