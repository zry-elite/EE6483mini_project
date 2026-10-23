import argparse
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "accuracy_comparison_3models.png"

BAR_WIDTH = 0.36
GROUP_GAP = 1.5
CAT_COLOR = "#D62728"
DOG_COLOR = "#2CA02C"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-class accuracy across model versions."
    )
    parser.add_argument(
        "--group",
        action="append",
        nargs=3,
        metavar=("LABEL", "CAT", "DOG"),
        help="Model version and its cat/dog accuracy in percent; repeat per model",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.group:
        groups = [(label, [float(cat), float(dog)]) for label, cat, dog in args.group]
    else:
        groups = [
            ("MobileNet_V3_small_baseline", [97.70, 97.80]),
            ("MobileNet_V3_small_RC0.7", [99.00, 98.20]),
            ("MobileNet_V3_small_RC0.5", [98.20, 98.50]),
        ]
    centers = [i * GROUP_GAP for i in range(len(groups))]
    labels = [
        f"{name}\n(overall {sum(values) / len(values):.2f}%)"
        for name, values in groups
    ]

    fig_width = max(6.5, 0.22 * max(len(name) for name, _ in groups) + 1.2 * len(groups))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    for center, (_, values) in zip(centers, groups):
        positions = [center - BAR_WIDTH / 2, center + BAR_WIDTH / 2]
        bars = ax.bar(positions, values, width=BAR_WIDTH, color=[CAT_COLOR, DOG_COLOR])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
            )

    ax.set_ylim(0, 105)
    ax.set_xlim(centers[0] - 0.8, centers[-1] + 0.8)
    ax.set_xticks(centers, labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison: MobileNetV3-Small Variants")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CAT_COLOR, DOG_COLOR)]
    ax.legend(handles, ["Cat", "Dog"], loc="lower right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
