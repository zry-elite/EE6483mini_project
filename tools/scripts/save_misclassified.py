import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model import build_model  # noqa: E402
from src.data import build_eval_transforms  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save misclassified validation images for one or more checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable, e.g. small=outputs/.../best.pth",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/base_1007"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/misclassified"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--norm-mean", nargs=3, type=float, default=(0.485, 0.456, 0.406))
    parser.add_argument("--norm-std", nargs=3, type=float, default=(0.229, 0.224, 0.225))
    return parser.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for images, _ in loader:
        logits = model(images.to(device))
        probs.append(logits.float().softmax(dim=1).cpu())
    return torch.cat(probs)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ds = datasets.ImageFolder(
        args.data_dir / args.split,
        transform=build_eval_transforms(
            mean=tuple(args.norm_mean), std=tuple(args.norm_std)
        ),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        classes = checkpoint["classes"]
        backbone = checkpoint["args"].get("backbone", "mobilenet_v3_small")
        model = build_model(
            num_classes=len(classes), pretrained=False, backbone=backbone
        ).to(device)
        model.load_state_dict(checkpoint["model"])

        for run in range(1, args.runs + 1):
            run_dir = args.output_dir / f"{label}_run{run}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)

            probs = predict(model, loader, device)
            preds = probs.argmax(dim=1)
            rows = []
            for i, (image_path, true_idx) in enumerate(ds.samples):
                pred_idx = int(preds[i])
                if pred_idx == true_idx:
                    continue
                confidence = float(probs[i, pred_idx])
                true_name = classes[true_idx]
                pred_name = classes[pred_idx]
                dst = run_dir / (
                    f"true-{true_name}_pred-{pred_name}_conf-{confidence:.3f}"
                    f"_{Path(image_path).name}"
                )
                shutil.copy2(image_path, dst)
                rows.append(
                    [Path(image_path).name, true_name, pred_name, f"{confidence:.4f}"]
                )

            with open(run_dir / "errors.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file", "true", "pred", "confidence"])
                writer.writerows(rows)

            acc = (preds.numpy() == [label_idx for _, label_idx in ds.samples]).mean()
            print(
                f"{label} run{run}: acc={acc:.4f} errors={len(rows)} -> {run_dir}"
            )


if __name__ == "__main__":
    main()
