from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
RESIZE_SIZE = 256
COLOR_JITTER_PARAMS = (0.2, 0.2, 0.2, 0.0)
ERASING_P = 0.25
ERASING_SCALE = (0.02, 0.15)
HFLIP_P = 0.5


def build_eval_transforms(
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_train_transforms(
    image_size: int = IMAGE_SIZE,
    augment: str = "none",
    rrc_scale: tuple[float, float] = (0.7, 1.0),
    color_jitter: tuple[float, float, float, float] = COLOR_JITTER_PARAMS,
    erasing_p: float = ERASING_P,
    erasing_scale: tuple[float, float] = ERASING_SCALE,
    hflip_p: float = HFLIP_P,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    ops = [op.strip() for op in augment.split(",") if op.strip()]
    unknown = set(ops) - {
        "none",
        "random_resized_crop",
        "color_jitter",
        "random_erasing",
        "horizontal_flip",
    }
    if unknown:
        raise ValueError(f"unknown augment: {sorted(unknown)}")
    if "none" in ops and len(ops) > 1:
        raise ValueError("'none' cannot be combined with other augmentations")

    if "random_resized_crop" in ops:
        spatial = transforms.RandomResizedCrop(
            image_size, scale=tuple(rrc_scale), ratio=(3 / 4, 4 / 3)
        )
    else:
        spatial = transforms.Compose(
            [transforms.Resize(RESIZE_SIZE), transforms.CenterCrop(image_size)]
        )

    geometric = [spatial]
    if "horizontal_flip" in ops:
        geometric.append(transforms.RandomHorizontalFlip(p=hflip_p))

    photometric = []
    if "color_jitter" in ops:
        brightness, contrast, saturation, hue = color_jitter
        photometric.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )

    tensor_ops = []
    if "random_erasing" in ops:
        tensor_ops.append(
            transforms.RandomErasing(
                p=erasing_p, scale=tuple(erasing_scale), value=0.0
            )
        )

    return transforms.Compose(
        [
            *geometric,
            *photometric,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            *tensor_ops,
        ]
    )


def build_eval_loader(
    data_dir: Path,
    split: str = "val",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[DataLoader, list[str]]:
    ds = datasets.ImageFolder(
        Path(data_dir) / split,
        transform=build_eval_transforms(image_size, mean, std),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, ds.classes


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
    augment: str = "none",
    rrc_scale: tuple[float, float] = (0.7, 1.0),
    color_jitter: tuple[float, float, float, float] = COLOR_JITTER_PARAMS,
    erasing_p: float = ERASING_P,
    erasing_scale: tuple[float, float] = ERASING_SCALE,
    hflip_p: float = HFLIP_P,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[DataLoader, DataLoader, list[str]]:
    data_dir = Path(data_dir)
    train_transform = build_train_transforms(
        image_size, augment, rrc_scale, tuple(color_jitter), erasing_p,
        tuple(erasing_scale), hflip_p, mean, std,
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)


    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader, _ = build_eval_loader(
        data_dir, "val", batch_size, num_workers, image_size, mean, std
    )
    return train_loader, val_loader, train_ds.classes
