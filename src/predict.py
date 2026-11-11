import argparse
import ast
import csv
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

from model import build_model
from src.data import build_eval_transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class FlatImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), index


def natural_key(path: Path):
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def parse_tuple(value, default):
    if isinstance(value, str):
        try:
            return tuple(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return default
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return default


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a flat folder of images with a binary checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--output", type=Path, default=Path("outputs/test_predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    saved_args = checkpoint.get("args", {})
    backbone = saved_args.get("backbone", "mobilenet_v3_small")
    mean = parse_tuple(saved_args.get("norm_mean"), (0.485, 0.456, 0.406))
    std = parse_tuple(saved_args.get("norm_std"), (0.229, 0.224, 0.225))

    model = build_model(
        num_classes=len(classes), pretrained=False, backbone=backbone
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    paths = sorted(
        (
            path
            for path in Path(args.data_dir).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ),
        key=natural_key,
    )
    if not paths:
        raise SystemExit(f"no images found in {args.data_dir}")

    dataset = FlatImageDataset(
        paths, build_eval_transforms(mean=mean, std=std)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    rows = []
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
            probs = logits.float().softmax(dim=1).cpu()
            for offset, index in enumerate(indices.tolist()):
                prob = probs[offset]
                pred = int(prob.argmax())
                rows.append(
                    (
                        paths[index].name,
                        classes[pred],
                        f"{prob[pred].item():.4f}",
                        f"{prob[classes.index('cat')].item():.4f}",
                        f"{prob[classes.index('dog')].item():.4f}",
                    )
                )

    rows.sort(key=lambda row: natural_key(Path(row[0])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class", "confidence", "prob_cat", "prob_dog"])
        writer.writerows(rows)

    counts = Counter(row[1] for row in rows)
    print(
        f"checkpoint epoch {checkpoint['epoch']} val_acc {checkpoint['val_acc']:.4f} "
        f"backbone={backbone} n={len(rows)}"
    )
    print(f"predictions: {dict(counts)}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
