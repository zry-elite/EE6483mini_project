import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGE_SIZE = 32
CACHE_NAME = "cifar10_cache.pt"


class CifarCacheDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int):
        image = self.images[index].permute(2, 0, 1)
        return self.transform(image), int(self.labels[index])


def load_cache(data_dir: Path):
    cache_path = Path(data_dir) / CACHE_NAME
    if not cache_path.exists():
        raise SystemExit(
            f"cache not found: {cache_path}; run tools/scripts/prepare_cifar10_cache.py"
        )
    return torch.load(cache_path, weights_only=True)


def stratified_split(
    targets: list[int], val_per_class: int, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    targets_arr = np.array(targets)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(targets_arr):
        indices = np.where(targets_arr == cls)[0].tolist()
        rng.shuffle(indices)
        val_idx.extend(indices[:val_per_class])
        train_idx.extend(indices[val_per_class:])
    return train_idx, val_idx


def compute_cifar_stats(
    data_dir: Path = Path("data/cifar10"),
    seed: int = 42,
    val_per_class: int = 500,
    workers: int = 0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cache = load_cache(data_dir)
    train_idx, _ = stratified_split(
        cache["train_labels"].tolist(), val_per_class, seed
    )
    index = torch.tensor(train_idx)
    pixel_sum = torch.zeros(3, dtype=torch.float64)
    pixel_sq = torch.zeros(3, dtype=torch.float64)
    count = 0
    for chunk in torch.split(index, 10000):
        pixels = cache["train_images"][chunk].double() / 255.0
        pixel_sum += pixels.sum(dim=(0, 1, 2))
        pixel_sq += (pixels * pixels).sum(dim=(0, 1, 2))
        count += pixels.shape[0] * IMAGE_SIZE * IMAGE_SIZE
    mean = pixel_sum / count
    std = (pixel_sq / count - mean * mean).sqrt()
    return tuple(round(v, 4) for v in mean.tolist()), tuple(
        round(v, 4) for v in std.tolist()
    )


def build_cifar10_eval_loader(
    data_dir: Path = Path("data/cifar10"),
    split: str = "test",
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    val_per_class: int = 500,
    mean: tuple[float, float, float] = CIFAR_MEAN,
    std: tuple[float, float, float] = CIFAR_STD,
) -> tuple[DataLoader, list[str]]:
    transform = v2.Compose(
        [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    )
    cache = load_cache(data_dir)
    if split == "test":
        ds = CifarCacheDataset(
            cache["test_images"], cache["test_labels"], transform
        )
    elif split == "val":
        full = CifarCacheDataset(
            cache["train_images"], cache["train_labels"], transform
        )
        _, val_idx = stratified_split(
            cache["train_labels"].tolist(), val_per_class, seed
        )
        ds = Subset(full, val_idx)
    else:
        raise ValueError(f"unknown split: {split}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, cache["classes"]


def build_cifar10_dataloaders(
    data_dir: Path = Path("data/cifar10"),
    batch_size: int = 128,
    num_workers: int = 6,
    seed: int = 42,
    val_per_class: int = 500,
    mean: tuple[float, float, float] = CIFAR_MEAN,
    std: tuple[float, float, float] = CIFAR_STD,
    rrc_scale: tuple[float, float] = (0.8, 1.0),
    erasing_p: float = 0.25,
    erasing_scale: tuple[float, float] = (0.02, 0.15),
    hflip_p: float = 0.5,
):
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(
                IMAGE_SIZE, scale=tuple(rrc_scale), ratio=(3 / 4, 4 / 3)
            ),
            v2.RandomHorizontalFlip(p=hflip_p),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
            v2.RandomErasing(
                p=erasing_p, scale=tuple(erasing_scale), value=0.0
            ),
        ]
    )
    eval_transform = v2.Compose(
        [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    )

    cache = load_cache(data_dir)
    train_full = CifarCacheDataset(
        cache["train_images"], cache["train_labels"], train_transform
    )
    val_full = CifarCacheDataset(
        cache["train_images"], cache["train_labels"], eval_transform
    )
    test_ds = CifarCacheDataset(
        cache["test_images"], cache["test_labels"], eval_transform
    )

    train_idx, val_idx = stratified_split(
        cache["train_labels"].tolist(), val_per_class, seed
    )
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        persistent_workers=num_workers > 0,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, persistent_workers=False, **common)
    test_loader = DataLoader(
        test_ds, shuffle=False, persistent_workers=False, **common
    )
    return train_loader, val_loader, test_loader, cache["classes"]
