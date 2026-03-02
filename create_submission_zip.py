from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def gather_root_files(repo_root: Path) -> list[Path]:
    keep_names = {
        "README.md",
        "requirements.txt",
        "pipeline_walkthrough.qmd",
        "pipeline_walkthrough.html",
        "phenology_config.py",
    }

    files: list[Path] = []

    for path in repo_root.glob("*.py"):
        files.append(path)

    for name in keep_names:
        path = repo_root / name
        if path.exists() and path.is_file() and path not in files:
            files.append(path)

    return sorted(files)


def gather_data_files(repo_root: Path, include_generated: bool) -> list[Path]:
    data_root = repo_root / "data"
    if not data_root.exists():
        return []

    include_dirs = [
        data_root / "blossoms",
        data_root / "noaa",
        data_root / "metadata",
    ]

    if include_generated:
        include_dirs.extend(
            [
                data_root / "model_inputs",
                data_root / "model_outputs",
            ]
        )

    files: list[Path] = []

    readme = data_root / "README.md"
    if readme.exists() and readme.is_file():
        files.append(readme)

    for folder in include_dirs:
        if not folder.exists() or not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                files.append(path)

    return sorted(set(files))


def make_zip(repo_root: Path, output_zip: Path, include_generated: bool) -> tuple[int, int]:
    files = gather_root_files(repo_root) + gather_data_files(repo_root, include_generated)

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    with ZipFile(output_zip, mode="w", compression=ZIP_DEFLATED) as zf:
        for file_path in files:
            arcname = file_path.relative_to(repo_root)
            zf.write(file_path, arcname=arcname)
            total_bytes += file_path.stat().st_size

    return len(files), total_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducibility zip with pipeline Python scripts and required data files."
        )
    )
    parser.add_argument(
        "--output",
        default="submission_repro_bundle.zip",
        help="Output zip file path (default: submission_repro_bundle.zip)",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Also include generated data/model artifacts in data/model_inputs and data/model_outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_zip = (repo_root / args.output).resolve()

    count, total_bytes = make_zip(
        repo_root=repo_root,
        output_zip=output_zip,
        include_generated=args.include_generated,
    )

    size_mb = total_bytes / (1024 * 1024)
    print(f"Created: {output_zip}")
    print(f"Files added: {count}")
    print(f"Approximate uncompressed size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
