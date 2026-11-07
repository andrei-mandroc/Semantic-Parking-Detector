"""Generate a PKLot manifest and data-quality report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.manifest import MANIFEST_FIELDS, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/PKLot"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/pklot_manifest.csv"))
    parser.add_argument("--report", type=Path, default=Path("outputs/pklot_manifest_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", help="Fallback camera when not present in file paths")
    parser.add_argument("--weather", help="Fallback weather when not present in file paths")
    args = parser.parse_args()

    rows, report = build_manifest(args.dataset_root, args.seed, args.camera, args.weather)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} valid pairs to {args.manifest}")
    print(f"Wrote data-quality report to {args.report}")
    print(f"Split counts: {report['by_split']}")


if __name__ == "__main__":
    main()
