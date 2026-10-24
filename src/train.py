import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def mixup_batch(images, targets, alpha):
    lam = random.betavariate(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, targets, targets[index], lam


def cutmix_batch(images, targets, alpha):
    lam = random.betavariate(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    _, _, height, width = images.shape
    cut_h = int(height * (1.0 - lam) ** 0.5)
    cut_w = int(width * (1.0 - lam) ** 0.5)
    center_y = random.randint(0, height - 1)
    center_x = random.randint(0, width - 1)
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (height * width)
    return mixed, targets, targets[index], lam


def soft_cross_entropy(logits, targets_a, targets_b, lam):
    log_probs = logits.float().log_softmax(dim=1)
    loss_a = log_probs.gather(1, targets_a[:, None]).squeeze(1)
    loss_b = log_probs.gather(1, targets_b[:, None]).squeeze(1)
    return -(lam * loss_a + (1.0 - lam) * loss_b).mean()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp,
    mixup_alpha=0.0,
    cutmix_alpha=0.0,
    mix_prob=0.0,
    teacher=None,
    distill_temperature=3.0,
    distill_alpha=0.5,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        targets_a = targets_b = None
        lam = 1.0
        if mix_prob > 0.0 and random.random() < mix_prob:
            use_mixup = random.random() < 0.5
            if use_mixup and mixup_alpha > 0.0:
                images, targets_a, targets_b, lam = mixup_batch(
                    images, targets, mixup_alpha
                )
            elif not use_mixup and cutmix_alpha > 0.0:
                images, targets_a, targets_b, lam = cutmix_batch(
                    images, targets, cutmix_alpha
                )

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(images)
        if targets_a is not None:
            loss = soft_cross_entropy(outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        if teacher is not None:
            with torch.no_grad(), autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                teacher_logits = teacher(images)
            temperature = distill_temperature
            teacher_probs = (teacher_logits.float() / temperature).softmax(dim=1)
            student_log_probs = (outputs.float() / temperature).log_softmax(dim=1)
            kd_loss = (
                F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                * temperature**2
            )
            loss = distill_alpha * loss + (1.0 - distill_alpha) * kd_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp, num_classes, tta="none"):
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

        preds = outputs.float().argmax(1)
        if tta == "flip":
            with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                flip_outputs = model(images.flip(-1))
            probs = (
                outputs.float().softmax(1) + flip_outputs.float().softmax(1)
            ) / 2
            preds = probs.argmax(1)

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
    parser.add_argument(
        "--backbone",
        choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"],
        default="mobilenet_v3_small",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--augment",
        default="none",
        help="Comma-separated augmentations: none | random_resized_crop | "
        "color_jitter | random_erasing | horizontal_flip",
    )
    parser.add_argument(
        "--rrc-scale",
        nargs=2,
        type=float,
        default=(0.7, 1.0),
        metavar=("MIN", "MAX"),
        help="RandomResizedCrop scale range",
    )
    parser.add_argument(
        "--color-jitter",
        nargs=4,
        type=float,
        default=(0.2, 0.2, 0.2, 0.0),
        metavar=("BRIGHTNESS", "CONTRAST", "SATURATION", "HUE"),
        help="ColorJitter magnitudes",
    )
    parser.add_argument(
        "--erasing-p",
        type=float,
        default=0.25,
        help="RandomErasing probability",
    )
    parser.add_argument(
        "--erasing-scale",
        nargs=2,
        type=float,
        default=(0.02, 0.15),
        metavar=("MIN", "MAX"),
        help="RandomErasing erased area ratio range",
    )
    parser.add_argument(
        "--hflip-p",
        type=float,
        default=0.5,
        help="RandomHorizontalFlip probability",
    )
    parser.add_argument(
        "--eval-tta",
        choices=["none", "flip"],
        default="none",
        help="Test-time augmentation used for validation",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Mixup Beta alpha (0 disables)",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=0.0,
        help="CutMix Beta alpha (0 disables)",
    )
    parser.add_argument(
        "--mix-prob",
        type=float,
        default=0.0,
        help="Probability of applying Mixup or CutMix per batch",
    )
    parser.add_argument(
        "--norm-mean",
        nargs=3,
        type=float,
        default=(0.485, 0.456, 0.406),
        metavar=("R", "G", "B"),
        help="Normalization mean",
    )
    parser.add_argument(
        "--norm-std",
        nargs=3,
        type=float,
        default=(0.229, 0.224, 0.225),
        metavar=("R", "G", "B"),
        help="Normalization std",
    )
    parser.add_argument(
        "--distill-checkpoint",
        type=Path,
        default=None,
        help="Teacher checkpoint for knowledge distillation",
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=3.0,
        help="Distillation softmax temperature",
    )
    parser.add_argument(
        "--distill-alpha",
        type=float,
        default=0.5,
        help="Weight of the hard-label loss (1-alpha goes to KD)",
    )
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
        args.data_dir,
        args.batch_size,
        args.workers,
        args.seed,
        augment=args.augment,
        rrc_scale=tuple(args.rrc_scale),
        color_jitter=tuple(args.color_jitter),
        erasing_p=args.erasing_p,
        erasing_scale=tuple(args.erasing_scale),
        hflip_p=args.hflip_p,
        mean=tuple(args.norm_mean),
        std=tuple(args.norm_std),
    )
    print(
        f"device={device} amp={amp} classes={classes} "
        f"augment={args.augment} rrc_scale={tuple(args.rrc_scale)} "
        f"color_jitter={tuple(args.color_jitter)} "
        f"erasing_p={args.erasing_p} erasing_scale={tuple(args.erasing_scale)} "
        f"hflip_p={args.hflip_p} eval_tta={args.eval_tta} "
        f"mixup_alpha={args.mixup_alpha} cutmix_alpha={args.cutmix_alpha} "
        f"mix_prob={args.mix_prob} norm_mean={tuple(args.norm_mean)} "
        f"norm_std={tuple(args.norm_std)}"
    )
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_model(
        num_classes=len(classes), pretrained=True, backbone=args.backbone
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"backbone={args.backbone} params={num_params:,}")

    teacher = None
    if args.distill_checkpoint is not None:
        teacher_ckpt = torch.load(
            args.distill_checkpoint, map_location=device, weights_only=True
        )
        teacher_backbone = teacher_ckpt["args"].get(
            "backbone", "mobilenet_v3_small"
        )
        teacher = build_model(
            num_classes=len(classes), pretrained=False, backbone=teacher_backbone
        ).to(device)
        teacher.load_state_dict(teacher_ckpt["model"])
        teacher.eval()
        teacher.requires_grad_(False)
        teacher_params = sum(p.numel() for p in teacher.parameters())
        print(
            f"teacher={teacher_backbone} params={teacher_params:,} "
            f"T={args.distill_temperature} alpha={args.distill_alpha}"
        )
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
                teacher=teacher,
                distill_temperature=args.distill_temperature,
                distill_alpha=args.distill_alpha,
            )
            val_loss, val_acc, _ = evaluate(
                model, val_loader, criterion, device, amp, len(classes),
                args.eval_tta,
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
        model, val_loader, criterion, device, amp, len(classes), args.eval_tta
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
