import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast

from model import build_model
from src.data import build_eval_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a labeled split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/base_1003"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tta", choices=["none", "flip"], default="none")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp, num_classes, tta):
    model.eval()
    plain_confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    tta_confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss = 0.0
    total = 0
    changed = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)

        probs = logits.float().softmax(1)
        if tta == "flip":
            with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                flip_logits = model(images.flip(-1))
            probs = (probs + flip_logits.float().softmax(1)) / 2

        plain_preds = logits.float().argmax(1)
        tta_preds = probs.argmax(1)
        changed += (plain_preds != tta_preds).sum().item()

        for t, p in zip(targets.cpu(), plain_preds.cpu()):
            plain_confusion[t, p] += 1
        for t, p in zip(targets.cpu(), tta_preds.cpu()):
            tta_confusion[t, p] += 1

    return total_loss / total, plain_confusion, tta_confusion, changed


def print_confusion(confusion, classes, title):
    header = "".join(f"{c:>8}" for c in classes)
    print(f"\n{title} (rows=true, cols=pred)\n{'':>6}{header}")
    for i, name in enumerate(classes):
        row = "".join(f"{int(confusion[i, j]):>8}" for j in range(len(classes)))
        support = int(confusion[i].sum())
        acc = confusion[i, i].item() / support if support else 0.0
        print(f"{name:>6}{row}   acc={acc:.4f}")


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    backbone = checkpoint.get("args", {}).get("backbone", "mobilenet_v3_small")
    model = build_model(
        num_classes=len(classes), pretrained=False, backbone=backbone
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    loader, _ = build_eval_loader(
        args.data_dir, args.split, args.batch_size, args.workers
    )
    criterion = nn.CrossEntropyLoss()

    print(
        f"checkpoint epoch {checkpoint['epoch']} "
        f"val_acc {checkpoint['val_acc']:.4f} | "
        f"split={args.split} n={len(loader.dataset)} tta={args.tta}"
    )
    loss, plain_confusion, tta_confusion, changed = evaluate(
        model, loader, criterion, device, amp, len(classes), args.tta
    )

    plain_acc = plain_confusion.diag().sum().item() / plain_confusion.sum().item()
    tta_acc = tta_confusion.diag().sum().item() / tta_confusion.sum().item()
    print(f"loss {loss:.4f} | plain acc {plain_acc:.4f}", end="")
    if args.tta == "flip":
        print(
            f" | flip tta acc {tta_acc:.4f} | preds changed {changed}"
        )
    else:
        print()

    print_confusion(tta_confusion, classes, "tta predictions")


if __name__ == "__main__":
    main()
