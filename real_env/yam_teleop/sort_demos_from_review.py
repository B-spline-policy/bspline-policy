#!/usr/bin/env python3
"""Create data/succ and data/fail, copy episode dirs from a reviewer JSON, then clear data/demos."""

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEMOS_DIR = DATA_DIR / "demos"


def _safe_demo_id(demo_id: str) -> bool:
    if not demo_id or demo_id != Path(demo_id).name:
        return False
    return ".." not in demo_id


def _clear_demos_dir(demos_dir: Path, dry_run: bool) -> None:
    """Remove every file and directory directly under demos_dir."""
    if not demos_dir.is_dir():
        return
    for child in sorted(demos_dir.iterdir(), key=lambda p: p.name):
        if dry_run:
            print(f"  remove {child}")
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy reviewed demos into data/succ and data/fail from review_results JSON, "
            "then remove everything under data/demos."
        )
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=DEMOS_DIR / "review_results_20260407_171358.json",
        help="Path to review_results_*.json (default: sample file under data/demos)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not create dirs or copy",
    )
    args = parser.parse_args()
    json_path = args.json_path.expanduser().resolve()
    if not json_path.is_file():
        print(f"Error: JSON not found: {json_path}", file=sys.stderr)
        return 1

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    success_files = data.get("success_files") or []
    fail_files = data.get("fail_files") or []
    if not isinstance(success_files, list) or not isinstance(fail_files, list):
        print("Error: JSON must contain list fields success_files and fail_files", file=sys.stderr)
        return 1

    succ_dir = DATA_DIR / "succ"
    fail_dir = DATA_DIR / "fail"

    if not args.dry_run:
        succ_dir.mkdir(parents=True, exist_ok=True)
        fail_dir.mkdir(parents=True, exist_ok=True)

    def copy_one(demo_id: str, dest_root: Path, label: str) -> None:
        if not isinstance(demo_id, str) or not _safe_demo_id(demo_id):
            print(f"  skip invalid id: {demo_id!r}", file=sys.stderr)
            return
        src = (DEMOS_DIR / demo_id).resolve()
        try:
            src.relative_to(DEMOS_DIR.resolve())
        except ValueError:
            print(f"  skip path escape: {demo_id}", file=sys.stderr)
            return
        if not src.is_dir():
            print(f"  missing source dir: {src}", file=sys.stderr)
            return
        dst = dest_root / demo_id
        if args.dry_run:
            print(f"  [{label}] {src} -> {dst}")
            return
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  [{label}] {demo_id}")

    print(f"JSON: {json_path}")
    print(f"Source: {DEMOS_DIR}")
    print(f"Success -> {succ_dir} ({len(success_files)} ids)")
    for demo_id in success_files:
        copy_one(demo_id, succ_dir, "succ")
    print(f"Fail -> {fail_dir} ({len(fail_files)} ids)")
    for demo_id in fail_files:
        copy_one(demo_id, fail_dir, "fail")

    print(f"Clear {DEMOS_DIR} (all entries)")
    _clear_demos_dir(DEMOS_DIR, args.dry_run)

    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
