import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = (
    PROJECT_ROOT / "outputs" / "base_1003_mobilenetv3_small" / "metrics.csv"
)

TRAIN_LOSS_COLOR = "#FF8C00"
TRAIN_ACC_COLOR = "#1F77B4"
VAL_LOSS_COLOR = "#D98C8C"
VAL_ACC_COLOR = "#2CA02C"


def load_metrics(path: Path) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            columns["epoch"].append(int(row["epoch"]))
            for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
                columns[key].append(float(row[key]))
    return columns


def plot_split(epochs, loss, acc, loss_color, acc_color, title, output_path):
    fig, ax_loss = plt.subplots(figsize=(8, 4.5))
    loss_line, = ax_loss.plot(
        epochs, loss, color=loss_color, linewidth=1.8, label="Loss"
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    ax_acc = ax_loss.twinx()
    acc_line, = ax_acc.plot(
        epochs, acc, color=acc_color, linewidth=1.8, label="Accuracy"
    )
    ax_acc.set_ylabel("Accuracy")
    ax_acc.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax_loss.set_title(title)
    ax_loss.legend(handles=[loss_line, acc_line], loc="center right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/val loss and accuracy curves.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to the directory containing --metrics",
    )
    args = parser.parse_args()

    if not args.metrics.is_file():
        raise SystemExit(f"metrics file not found: {args.metrics}")

    output_dir = args.output_dir or args.metrics.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_metrics(args.metrics)
    epochs = data["epoch"]

    train_path = output_dir / "train_curves.png"
    val_path = output_dir / "val_curves.png"

    plot_split(
        epochs,
        data["train_loss"],
        data["train_acc"],
        TRAIN_LOSS_COLOR,
        TRAIN_ACC_COLOR,
        "Training Loss & Accuracy",
        train_path,
    )
    plot_split(
        epochs,
        data["val_loss"],
        data["val_acc"],
        VAL_LOSS_COLOR,
        VAL_ACC_COLOR,
        "Validation Loss & Accuracy",
        val_path,
    )

    print(f"saved {train_path}")
    print(f"saved {val_path}")


if __name__ == "__main__":
    main()
