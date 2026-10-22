import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASSES = ("cat", "dog")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "dataset" / "base_1003"


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced cat/dog subset.")
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-train", type=int, default=8000, help="images per class")
    parser.add_argument("--n-val", type=int, default=1000, help="images per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        if not args.force:
            raise SystemExit(f"output already exists: {args.out_dir} (use --force to overwrite)")
        shutil.rmtree(args.out_dir)

    rng = random.Random(args.seed)
    plan = {"train": args.n_train, "val": args.n_val}

    for split, per_class in plan.items():
        for cls in CLASSES:
            src = args.src_dir / split / cls
            if not src.is_dir():
                raise SystemExit(f"source directory not found: {src}")

            images = list_images(src)
            if len(images) < per_class:
                raise SystemExit(
                    f"not enough images in {src}: {len(images)} < {per_class}"
                )

            selected = rng.sample(images, per_class)
            dst = args.out_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for image in selected:
                shutil.copy2(image, dst / image.name)
            print(f"{split}/{cls}: {len(selected)} images -> {dst}")

    print(f"dataset ready: {args.out_dir}")


if __name__ == "__main__":
    main()
