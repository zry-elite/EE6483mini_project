import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "cifar10"


def load_split(root: Path, split: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    ds = datasets.ImageFolder(root / split)
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
    parser = argparse.ArgumentParser(
        description="Cache an ImageFolder-style CIFAR dataset in memory."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = Path(args.data_dir)
    output = args.output or root / "cifar10_cache.pt"

    payload = {}
    classes = None
    for split in ("train", "val", "test"):
        if not (root / split).is_dir():
            continue
        images, labels, split_classes = load_split(root, split)
        if classes is None:
            classes = split_classes
        elif split_classes != classes:
            raise SystemExit(
                f"class mismatch in {split}: {split_classes} vs {classes}"
            )
        payload[f"{split}_images"] = images
        payload[f"{split}_labels"] = labels
        print(f"{split}: {len(images)} images")

    if classes is None:
        raise SystemExit(f"no train/val/test split found under {root}")
    payload["classes"] = classes
    torch.save(payload, output)
    print(f"classes={classes} -> {output}")


if __name__ == "__main__":
    main()
