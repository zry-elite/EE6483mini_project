from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CIFAR_ROOT = PROJECT_ROOT / "data" / "cifar10"
CACHE_PATH = CIFAR_ROOT / "cifar10_cache.pt"


def load_split(split: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    ds = datasets.ImageFolder(CIFAR_ROOT / split)
    images = np.zeros((len(ds), 32, 32, 3), dtype=np.uint8)
    labels = np.zeros(len(ds), dtype=np.int64)
    for i, (path, label) in enumerate(ds.samples):
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        if array.shape != (32, 32, 3):
            array = np.asarray(image.resize((32, 32)), dtype=np.uint8)
        images[i] = array
        labels[i] = label
    return torch.from_numpy(images), torch.from_numpy(labels), ds.classes


def main() -> None:
    train_images, train_labels, classes = load_split("train")
    test_images, test_labels, test_classes = load_split("test")
    if classes != test_classes:
        raise SystemExit(f"class mismatch: {classes} vs {test_classes}")

    torch.save(
        {
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
            "classes": classes,
        },
        CACHE_PATH,
    )
    print(
        f"cached train={len(train_images)} test={len(test_images)} "
        f"classes={classes} -> {CACHE_PATH}"
    )


if __name__ == "__main__":
    main()
