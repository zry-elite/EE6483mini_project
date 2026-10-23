import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def collate_images(batch):
    return [image for image, _ in batch]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std of an ImageFolder split."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "base_1007" / "train",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    ds = datasets.ImageFolder(args.data_dir, transform=transforms.ToTensor())
    loader = DataLoader(
        ds,
        batch_size=64,
        num_workers=args.workers,
        shuffle=False,
        collate_fn=collate_images,
    )

    pixel_sum = torch.zeros(3, dtype=torch.float64)
    pixel_sq = torch.zeros(3, dtype=torch.float64)
    count = 0
    for images in loader:
        for image in images:
            image = image.double()
            pixel_sum += image.sum(dim=(1, 2))
            pixel_sq += (image * image).sum(dim=(1, 2))
            count += image.shape[1] * image.shape[2]

    mean = pixel_sum / count
    var = pixel_sq / count - mean * mean
    std = var.sqrt()

    print(f"images={len(ds)} pixels_per_channel={count}")
    print(f"mean {[round(v, 4) for v in mean.tolist()]}")
    print(f"std  {[round(v, 4) for v in std.tolist()]}")
    print("--norm-mean " + " ".join(f"{v:.4f}" for v in mean.tolist()))
    print("--norm-std  " + " ".join(f"{v:.4f}" for v in std.tolist()))


if __name__ == "__main__":
    main()
