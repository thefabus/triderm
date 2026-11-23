"""Wound photos with one binary mask per wound, and the train/eval transforms."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as TF
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MASK_PATTERN = re.compile(r"(\d+[a-zA-Z]?)_wounds_(\d+)")
PHOTO_PATTERN = re.compile(r"(\d+[a-zA-Z]?)")


@dataclass(frozen=True)
class WoundSample:
    """One wound: its photo and its binary mask."""

    photo_path: Path
    mask_path: Path
    photo_id: str
    wound_id: int


def find_wound_samples(photos_dir: Path, masks_dir: Path, exclude: set[str]) -> list[WoundSample]:
    """Pair each mask `<photo_id>_wounds_<k>.<ext>` with the photo whose name starts with the ID."""
    photos: dict[str, Path] = {}
    for p in sorted(photos_dir.iterdir()):
        match = PHOTO_PATTERN.search(p.name)
        if p.suffix.lower() in IMAGE_SUFFIXES and match:
            photos.setdefault(match.group(1), p)

    samples = []
    for mask in sorted(masks_dir.iterdir()):
        match = MASK_PATTERN.search(mask.name)
        if mask.suffix.lower() not in IMAGE_SUFFIXES or not match:
            continue
        photo_id, wound_id = match.group(1), int(match.group(2))
        if photo_id in exclude:
            continue
        if photo_id not in photos:
            print(f"Warning: no photo for mask {mask.name}")
            continue
        samples.append(WoundSample(photos[photo_id], mask, photo_id, wound_id))
    if not samples:
        raise ValueError(f"No photo/mask pairs found in {photos_dir} and {masks_dir}")
    return samples


def _normalize(image: torch.Tensor, imagenet_normalize: bool) -> torch.Tensor:
    if imagenet_normalize:
        return TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return torch.clamp(image, 0, 1)


class EvalTransform:
    """Resize the short side to size+3, then crop size x size centred on the wound mask."""

    def __init__(self, size: int, imagenet_normalize: bool) -> None:
        self.size = size
        self.imagenet_normalize = imagenet_normalize

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        img = TF.pil_to_tensor(image).float() / 255.0
        msk = TF.pil_to_tensor(mask).squeeze(0).long()
        _, h, w = img.shape
        scale = (self.size + 3) / min(h, w)
        new_hw = [int(h * scale), int(w * scale)]
        img = TF.resize(img, new_hw)
        msk = TF.resize(msk.unsqueeze(0).float(), new_hw).squeeze(0).long()

        coords = (msk > 0).nonzero(as_tuple=False)
        if coords.numel():
            yc = int(coords[:, 0].min() + coords[:, 0].max()) // 2
            xc = int(coords[:, 1].min() + coords[:, 1].max()) // 2
        else:
            yc = xc = self.size // 2
        bottom = min(max(0, yc - self.size // 2) + self.size, img.shape[1])
        right = min(max(0, xc - self.size // 2) + self.size, img.shape[2])
        top, left = bottom - self.size, right - self.size
        img = img[:, top:bottom, left:right]
        msk = msk[top:bottom, left:right]
        return _normalize(img, self.imagenet_normalize), msk


class TrainTransform:
    """Random non-empty wound crop plus geometric, colour, noise and blur augmentations."""

    def __init__(self, size: int, imagenet_normalize: bool) -> None:
        self.imagenet_normalize = imagenet_normalize
        self.aug = A.Compose(
            [
                A.LongestMaxSize(max_size=4 * size),
                A.CropNonEmptyMaskIfExists(height=2 * size, width=2 * size, ignore_values=[0]),
                A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=40, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=40, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
                A.ToGray(p=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=7, sigma_limit=(0.1, 2.0), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                ToTensorV2(),
            ]
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.aug(image=np.array(image), mask=np.array(mask).squeeze())
        img = out["image"].float() / 255.0
        return _normalize(img, self.imagenet_normalize), out["mask"].long()


class WoundDataset(Dataset):
    """Items: (image, mask, photo_id, wound_id); with `two_views`, a second augmented view."""

    def __init__(
        self,
        samples: list[WoundSample],
        transform: EvalTransform | TrainTransform,
        two_views: bool = False,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.two_views = two_views

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        s = self.samples[idx]
        photo = ImageOps.exif_transpose(Image.open(s.photo_path).convert("RGB"))
        mask = Image.fromarray(np.array(Image.open(s.mask_path).convert("L")))
        image, msk = self.transform(photo, mask)
        if not self.two_views:
            return image, msk, s.photo_id, s.wound_id
        image2, msk2 = self.transform(photo, mask)
        return image, msk, s.photo_id, s.wound_id, image2, msk2
