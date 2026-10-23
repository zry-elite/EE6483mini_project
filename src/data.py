from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
RESIZE_SIZE = 256


def build_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader, list[str]]:
    data_dir = Path(data_dir)
    transform = build_transforms(image_size)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_ds.classes
