import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset

from model import build_model
from src.cifar import build_cifar10_dataloaders, compute_cifar_stats
from src.train import evaluate, save_checkpoint, set_seed, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 training.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cifar10_mobilenetv3_large"),
    )
    parser.add_argument(
        "--backbone",
        choices=["mobilenet_v3_large_cifar"],
        default="mobilenet_v3_large_cifar",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rrc-scale",
        nargs=2,
        type=float,
        default=(0.8, 1.0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--erasing-p", type=float, default=0.25)
    parser.add_argument(
        "--erasing-scale",
        nargs=2,
        type=float,
        default=(0.02, 0.15),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument(
        "--norm-mean", nargs=3, type=float, default=None, metavar=("R", "G", "B")
    )
    parser.add_argument(
        "--norm-std", nargs=3, type=float, default=None, metavar=("R", "G", "B")
    )
    parser.add_argument(
        "--class-weights",
        choices=["none", "inverse", "effective"],
        default="none",
        help="class weighting for the training loss",
    )
    parser.add_argument(
        "--effective-beta",
        type=float,
        default=0.9999,
        help="beta for effective-number class weights",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="focal loss gamma (0 disables focal, uses weighted CE)",
    )
    return parser.parse_args()


def get_train_labels(loader) -> torch.Tensor:
    ds = loader.dataset
    if isinstance(ds, Subset):
        return ds.dataset.labels[torch.tensor(ds.indices)]
    return ds.labels


def compute_class_weights(
    labels: torch.Tensor, num_classes: int, mode: str, beta: float
) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "effective":
        weights = (1.0 - beta) / (1.0 - beta**counts)
    else:
        raise ValueError(f"unknown class weight mode: {mode}")
    return weights * (counts.sum() / (weights * counts).sum())


def print_confusion(confusion, classes, title):
    header = "".join(f"{c:>10}" for c in classes)
    print(f"\n{title} (rows=true, cols=pred)\n{'':>10}{header}")
    for i, name in enumerate(classes):
        row = "".join(f"{int(confusion[i, j]):>10}" for j in range(len(classes)))
        support = int(confusion[i].sum())
        acc = confusion[i, i].item() / support if support else 0.0
        print(f"{name:>10}{row}   acc={acc:.4f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"

    if args.norm_mean is None or args.norm_std is None:
        mean, std = compute_cifar_stats(
            args.data_dir, args.seed, args.val_per_class, args.workers
        )
    else:
        mean, std = tuple(args.norm_mean), tuple(args.norm_std)

    train_loader, val_loader, test_loader, classes = build_cifar10_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
        val_per_class=args.val_per_class,
        mean=mean,
        std=std,
        rrc_scale=tuple(args.rrc_scale),
        erasing_p=args.erasing_p,
        erasing_scale=tuple(args.erasing_scale),
    )

    print(
        f"device={device} amp={amp} classes={classes} "
        f"norm_mean={mean} norm_std={std}"
    )
    print(
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}"
    )

    model = build_model(
        num_classes=len(classes), pretrained=True, backbone=args.backbone
    ).to(device)
    print(
        f"backbone={args.backbone} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    class_weights = None
    if args.class_weights != "none":
        train_labels = get_train_labels(train_loader)
        counts = torch.bincount(train_labels, minlength=len(classes))
        class_weights = compute_class_weights(
            train_labels, len(classes), args.class_weights, args.effective_beta
        ).to(device)
        print(
            f"class_weights={args.class_weights} beta={args.effective_beta}\n"
            f"  counts={counts.tolist()}\n"
            f"  weights={[round(w, 4) for w in class_weights.tolist()]}"
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    eval_criterion = nn.CrossEntropyLoss()
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
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                amp,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                mix_prob=args.mix_prob,
                class_weights=class_weights,
                focal_gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            val_loss, val_acc, _ = evaluate(
                model, val_loader, eval_criterion, device, amp, len(classes)
            )
            scheduler.step()

            elapsed = time.time() - start
            writer.writerow(
                [epoch, train_loss, train_acc, val_loss, val_acc, lr, elapsed]
            )
            f.flush()

            print(
                f"epoch {epoch:03d}/{args.epochs} | "
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

    val_loss, val_acc, val_confusion = evaluate(
        model, val_loader, eval_criterion, device, amp, len(classes)
    )
    test_loss, test_acc, test_confusion = evaluate(
        model, test_loader, eval_criterion, device, amp, len(classes)
    )

    print(f"\nbest epoch {best_epoch} | best val acc {best_acc:.4f}")
    print(f"best checkpoint val acc {val_acc:.4f} loss {val_loss:.4f}")
    print(f"test acc {test_acc:.4f} loss {test_loss:.4f}")
    print_confusion(val_confusion, classes, "val confusion")
    print_confusion(test_confusion, classes, "test confusion")

    report_path = args.output_dir / "test_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["best_epoch", best_epoch])
        writer.writerow(["val_acc", f"{val_acc:.4f}"])
        writer.writerow(["val_loss", f"{val_loss:.4f}"])
        writer.writerow(["test_acc", f"{test_acc:.4f}"])
        writer.writerow(["test_loss", f"{test_loss:.4f}"])
        for i, name in enumerate(classes):
            support = int(test_confusion[i].sum())
            acc = test_confusion[i, i].item() / support if support else 0.0
            writer.writerow([f"test_acc_{name}", f"{acc:.4f}"])


if __name__ == "__main__":
    main()
