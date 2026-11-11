import argparse
import random
import shutil
from pathlib import Path

from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "data" / "cifar10"
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "CIFAR10_imbalance"
DEFAULT_MINORITY = "dog,frog,horse,ship,truck"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an imbalanced CIFAR-10 training set."
    )
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--minority-classes",
        default=DEFAULT_MINORITY,
        help="comma-separated class names",
    )
    parser.add_argument("--majority-per-class", type=int, default=4500)
    parser.add_argument("--minority-per-class", type=int, default=500)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        if not args.force:
            raise SystemExit(
                f"output already exists: {args.out_dir} (use --force to overwrite)"
            )
        shutil.rmtree(args.out_dir)

    rng = random.Random(args.seed)
    minority = {name.strip() for name in args.minority_classes.split(",")}

    train_src = datasets.ImageFolder(args.src_dir / "train")
    test_src = datasets.ImageFolder(args.src_dir / "test")
    classes = train_src.classes
    if minority - set(classes):
        raise SystemExit(f"unknown minority classes: {minority - set(classes)}")

    stats = {}
    for split in ("train", "val"):
        stats[split] = {}

    for cls in classes:
        by_class = [p for p, label in train_src.samples if classes[label] == cls]
        rng.shuffle(by_class)
        val_files = by_class[: args.val_per_class]
        rest = by_class[args.val_per_class :]

        target = (
            args.minority_per_class if cls in minority else args.majority_per_class
        )
        if len(rest) < target:
            raise SystemExit(f"not enough images for {cls}: {len(rest)} < {target}")
        train_files = rest[:target]

        for split, files in (("train", train_files), ("val", val_files)):
            dst = args.out_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for path in files:
                shutil.copy2(path, dst / Path(path).name)
            stats[split][cls] = len(files)

    for cls in classes:
        src = args.src_dir / "test" / cls
        dst = args.out_dir / "test" / cls
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in src.iterdir() if p.is_file())
        for path in files:
            shutil.copy2(path, dst / path.name)
        stats["test"] = stats.get("test", {})
        stats["test"][cls] = len(files)

    for split, counts in stats.items():
        print(f"{split}: {counts} total={sum(counts.values())}")
    ratio = args.majority_per_class / args.minority_per_class
    print(f"imbalance ratio (majority/minority per class) = {ratio:.1f}:1")
    print(f"dataset ready: {args.out_dir}")


if __name__ == "__main__":
    main()
