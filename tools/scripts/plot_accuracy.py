import argparse
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "base_1003_mobilenetv3_small" / "accuracy_bar.png"
)

BAR_WIDTH = 0.3
CAT_COLOR = "#D62728"
DOG_COLOR = "#2CA02C"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bar chart of per-class accuracy.")
    parser.add_argument(
        "--cat-acc", type=float, default=97.70, help="cat accuracy in percent"
    )
    parser.add_argument(
        "--dog-acc", type=float, default=97.80, help="dog accuracy in percent"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    values = [args.cat_acc, args.dog_acc]
    colors = [CAT_COLOR, DOG_COLOR]
    positions = [-BAR_WIDTH / 2, BAR_WIDTH / 2]

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(positions, values, width=BAR_WIDTH, color=colors)

    ax.set_ylim(0, 105)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks([0], ["Accuracy"])
    ax.set_title("MobileNet_V3_small_baseline")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, ["Cat", "Dog"], loc="lower right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
