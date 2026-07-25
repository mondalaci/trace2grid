"""Dataset + augmentation for rectified tool masks."""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import TRAIN_LONG_SIDE


def letterbox(
    image: np.ndarray,
    mask: np.ndarray,
    long_side: int,
    *,
    image_fill: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    pad_y = long_side - nh
    pad_x = long_side - nw
    top, left = pad_y // 2, pad_x // 2
    bottom, right = pad_y - top, pad_x - left
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=image_fill
    )
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return image, mask


def train_augment(*, image_fill: int = 255) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                shear=(-5, 5),
                border_mode=cv2.BORDER_CONSTANT,
                fill=image_fill,
                fill_mask=0,
                p=0.7,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(0.25, 0.25),
                    A.RandomGamma(gamma_limit=(70, 130)),
                    A.CLAHE(clip_limit=2.0),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.08)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4)),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=(3, 5)),
                ],
                p=0.3,
            ),
            # Synthetic soft contact-shadow-ish darkening near tools.
            A.RandomShadow(
                shadow_roi=(0, 0.2, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                p=0.45,
            ),
        ]
    )


class ToolSegDataset(Dataset):
    def __init__(
        self,
        root: Path,
        ids: list[str],
        *,
        augment: bool,
        long_side: int = TRAIN_LONG_SIDE,
        repeats: int = 1,
        image_fill: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.root = root
        self.ids = ids
        self.image_fill = image_fill
        fill_scalar = int(image_fill[0])
        self.augment = train_augment(image_fill=fill_scalar) if augment else None
        self.long_side = long_side
        self.repeats = max(1, repeats)

    def __len__(self) -> int:
        return len(self.ids) * self.repeats

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_id = self.ids[index % len(self.ids)]
        folder = self.root / sample_id
        image = cv2.imread(str(folder / "image.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(folder / "mask.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(folder)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, mask = letterbox(image, mask, self.long_side, image_fill=self.image_fill)
        if self.augment is not None:
            out = self.augment(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        # ImageNet-ish normalize helps transfer; keep simple [0,1] centered.
        x = image.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()
        y = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return {"image": x, "mask": y, "id": sample_id}


def load_manifest(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "manifest.json"
    return json.loads(path.read_text())
