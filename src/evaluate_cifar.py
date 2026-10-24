import argparse
import csv
from pathlib import Path

import torch
from torch.amp import autocast

from model import build_model
from src.cifar import build_cifar10_eval_loader, compute_cifar_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-class precision/recall/F1 for CIFAR-10."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def collect_confusion(model, loader, device, amp, num_classes):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        preds = logits.float().argmax(dim=1).cpu()
        for t, p in zip(targets, preds):
            confusion[t, p] += 1
    return confusion


def compute_metrics(confusion: torch.Tensor):
    num_classes = confusion.size(0)
    support = confusion.sum(dim=1)
    tp = confusion.diag()
    predicted = confusion.sum(dim=0)

    precision = torch.where(
        predicted > 0, tp / predicted.clamp(min=1), torch.zeros(num_classes)
    )
    recall = torch.where(
        support > 0, tp / support.clamp(min=1), torch.zeros(num_classes)
    )
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall).clamp(min=1e-12),
        torch.zeros(num_classes),
    )

    total = support.sum().item()
    accuracy = tp.sum().item() / total if total else 0.0
    macro = {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }
    weights = support.float() / total if total else torch.zeros(num_classes)
    weighted = {
        "precision": (precision * weights).sum().item(),
        "recall": (recall * weights).sum().item(),
        "f1": (f1 * weights).sum().item(),
    }
    return {
        "support": support.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "accuracy": accuracy,
        "macro": macro,
        "weighted": weighted,
    }


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
    backbone = saved_args.get("backbone", "mobilenet_v3_large_cifar")
    seed = int(saved_args.get("seed", 42))
    val_per_class = int(saved_args.get("val_per_class", 500))

    model = build_model(
        num_classes=len(classes), pretrained=False, backbone=backbone
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    mean, std = compute_cifar_stats(args.data_dir, seed, val_per_class)
    loader, loader_classes = build_cifar10_eval_loader(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=seed,
        val_per_class=val_per_class,
        mean=mean,
        std=std,
    )
    if loader_classes != classes:
        raise SystemExit(f"class mismatch: {loader_classes} vs {classes}")

    confusion = collect_confusion(model, loader, device, amp, len(classes))
    metrics = compute_metrics(confusion)

    print(
        f"checkpoint epoch {checkpoint['epoch']} split={args.split} "
        f"n={int(confusion.sum())} acc={metrics['accuracy']:.4f}"
    )
    print(f"{'class':>12}{'support':>9}{'precision':>11}{'recall':>9}{'f1':>9}")
    for i, name in enumerate(classes):
        print(
            f"{name:>12}{metrics['support'][i]:>9}"
            f"{metrics['precision'][i]:>11.4f}"
            f"{metrics['recall'][i]:>9.4f}"
            f"{metrics['f1'][i]:>9.4f}"
        )
    print(
        f"{'macro avg':>12}{sum(metrics['support']):>9}"
        f"{metrics['macro']['precision']:>11.4f}"
        f"{metrics['macro']['recall']:>9.4f}"
        f"{metrics['macro']['f1']:>9.4f}"
    )
    print(
        f"{'weighted avg':>12}{sum(metrics['support']):>9}"
        f"{metrics['weighted']['precision']:>11.4f}"
        f"{metrics['weighted']['recall']:>9.4f}"
        f"{metrics['weighted']['f1']:>9.4f}"
    )

    output = args.output or args.checkpoint.parent / (
        f"{args.split}_per_class_metrics.csv"
    )
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "support", "precision", "recall", "f1"])
        for i, name in enumerate(classes):
            writer.writerow(
                [
                    name,
                    metrics["support"][i],
                    f"{metrics['precision'][i]:.4f}",
                    f"{metrics['recall'][i]:.4f}",
                    f"{metrics['f1'][i]:.4f}",
                ]
            )
        for label, key in (("macro_avg", "macro"), ("weighted_avg", "weighted")):
            writer.writerow(
                [
                    label,
                    sum(metrics["support"]),
                    f"{metrics[key]['precision']:.4f}",
                    f"{metrics[key]['recall']:.4f}",
                    f"{metrics[key]['f1']:.4f}",
                ]
            )
        writer.writerow(["accuracy", "", "", "", f"{metrics['accuracy']:.4f}"])
    print(f"saved {output}")


if __name__ == "__main__":
    main()
