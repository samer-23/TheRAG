"""Run every connector for the Space Apps "Interplanetary Survival Guide: Martian Map"
resources and (re)build data/processed/documents.jsonl.

Usage:
    python ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "connectors"))

from common import PROCESSED_DIR  # noqa: E402

import trek  # noqa: E402
import usgs_hub  # noqa: E402
import mars2020_raw  # noqa: E402
import science_nasa  # noqa: E402

CONNECTORS = [
    ("NASA Mars Trek (layers, landing sites, nomenclature)", trek.run),
    ("USGS Astrogeology Hub (interactive Mars maps)", usgs_hub.run),
    ("Mars 2020 Perseverance raw images", mars2020_raw.run),
    ("NASA Science: Curiosity's path up Mount Sharp", science_nasa.run),
]


def main() -> int:
    out_path = PROCESSED_DIR / "documents.jsonl"
    if out_path.exists():
        out_path.unlink()  # start clean each run; connectors append

    total = 0
    for label, fn in CONNECTORS:
        print(f"\n=== {label} ===")
        try:
            total += len(fn())
        except Exception as e:  # keep going even if one source has an outage
            print(f"  ! {label} failed: {e}")

    print(f"\nDone. {total} documents written to {out_path}")
    return total


if __name__ == "__main__":
    main()
