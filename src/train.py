import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import build_model
from src.data import build_dataloaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp, num_classes):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        preds = outputs.argmax(1)
        total_loss += loss.item() * targets.size(0)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        for t, p in zip(targets.cpu(), preds.cpu()):
            confusion[t, p] += 1

    return total_loss / total, correct / total, confusion


def save_checkpoint(path, model, optimizer, epoch, val_acc, args, classes):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "args": {k: str(v) for k, v in vars(args).items()},
            "classes": classes,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline training on base_1003.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/base_1003"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/base_1003_mobilenetv3_small")
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"

    train_loader, val_loader, classes = build_dataloaders(
        args.data_dir, args.batch_size, args.workers, args.seed
    )
    print(f"device={device} amp={amp} classes={classes}")
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_model(num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device.type, enabled=amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pth"
    last_path = args.output_dir / "last.pth"
    metrics_path = args.output_dir / "metrics.csv"

    best_acc = 0.0
    best_epoch = 0

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]
        )

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, amp
            )
            val_loss, val_acc, _ = evaluate(
                model, val_loader, criterion, device, amp, len(classes)
            )
            scheduler.step()

            elapsed = time.time() - start
            writer.writerow(
                [epoch, train_loss, train_acc, val_loss, val_acc, lr, elapsed]
            )
            f.flush()

            print(
                f"epoch {epoch:02d}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"lr {lr:.2e} | {elapsed:.1f}s"
            )

            save_checkpoint(last_path, model, optimizer, epoch, val_acc, args, classes)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                save_checkpoint(
                    best_path, model, optimizer, epoch, val_acc, args, classes
                )

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    val_loss, val_acc, confusion = evaluate(
        model, val_loader, criterion, device, amp, len(classes)
    )

    print(f"\nbest epoch {best_epoch} | best val acc {best_acc:.4f}")
    print(f"best checkpoint val acc {val_acc:.4f} loss {val_loss:.4f}")

    header = "".join(f"{c:>8}" for c in classes)
    print(f"\nconfusion matrix (rows=true, cols=pred)\n{'':>6}{header}")
    for i, name in enumerate(classes):
        row = "".join(f"{int(confusion[i, j]):>8}" for j in range(len(classes)))
        support = int(confusion[i].sum())
        acc = confusion[i, i].item() / support if support else 0.0
        print(f"{name:>6}{row}   acc={acc:.4f}")


if __name__ == "__main__":
    main()
