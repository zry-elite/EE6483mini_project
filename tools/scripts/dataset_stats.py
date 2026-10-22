import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def count_images(directory: Path) -> int:
    return sum(
        1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def split_rows(split_dir: Path) -> list[tuple[str, int]]:
    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if class_dirs:
        return [(d.name, count_images(d)) for d in class_dirs]
    return [("(unlabeled)", count_images(split_dir))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Count cat/dog images per split.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"data directory not found: {data_dir}")

    split_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not split_dirs:
        raise SystemExit(f"no splits found under: {data_dir}")

    print(f"{'Split':<8}{'Class':<12}{'Count':>8}{'Share':>9}")
    print("-" * 37)

    grand_total = 0
    for split_dir in split_dirs:
        rows = split_rows(split_dir)
        split_total = sum(n for _, n in rows)
        labeled = not (len(rows) == 1 and rows[0][0] == "(unlabeled)")

        for name, count in rows:
            share = f"{count / split_total:>8.1%}" if labeled and split_total else ""
            print(f"{split_dir.name:<8}{name:<12}{count:>8}{share:>9}")

        print(f"{split_dir.name:<8}{'total':<12}{split_total:>8}")
        print("-" * 37)
        grand_total += split_total

    print(f"{'ALL':<8}{'total':<12}{grand_total:>8}")


if __name__ == "__main__":
    main()
