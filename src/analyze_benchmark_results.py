"""Create a concise analysis from benchmark_robustness.py summary.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


NUMERIC_FIELDS = {
    "strength",
    "num_samples",
    "auc",
    "tpr_0_1fpr",
    "tpr_1fpr",
    "id_top1",
    "id_top5",
    "bit_acc",
    "psnr",
    "ssim",
    "lpips",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in NUMERIC_FIELDS:
            if row.get(field) not in {None, ""}:
                row[field] = float(row[field])
    return rows


def mean(rows: list[dict], field: str) -> float:
    return float(np.mean([row[field] for row in rows]))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.result_dir / "summary.csv")
    methods = list(dict.fromkeys(row["method"] for row in rows))
    lines = [
        "# Detailed Benchmark Analysis",
        "",
        "This is a fixed-key pilot. HSTR and HSQR are reported separately as HSFW variants.",
        "",
        "## Attack-Family Ranking",
        "",
        "| Suite | Metric | Ranking (best to worst) |",
        "|---|---|---|",
    ]

    for suite in ["waves_core", "stirmark_compatible"]:
        for metric in ["auc", "tpr_1fpr", "id_top1", "bit_acc"]:
            values = []
            for method in methods:
                subset = [
                    row for row in rows if row["method"] == method and row["suite"] == suite
                ]
                if subset:
                    values.append((method, mean(subset, metric)))
            values.sort(key=lambda item: item[1], reverse=True)
            ranking = " > ".join(f"{method} ({value:.3f})" for method, value in values)
            lines.append(f"| {suite} | {metric} | {ranking} |")

    lines.extend(["", "## Worst Conditions by Method", ""])
    for method in methods:
        attacked = [row for row in rows if row["method"] == method and row["suite"] != "baseline"]
        worst_verify = sorted(attacked, key=lambda row: (row["tpr_1fpr"], row["auc"]))[:5]
        worst_identify = sorted(attacked, key=lambda row: (row["id_top1"], row["bit_acc"]))[:5]
        lines.extend(
            [
                f"### {method}",
                "",
                "Verification: "
                + ", ".join(
                    f"{row['name']} (TPR={row['tpr_1fpr']:.3f}, AUC={row['auc']:.3f})"
                    for row in worst_verify
                ),
                "",
                "Identification: "
                + ", ".join(
                    f"{row['name']} (Top-1={row['id_top1']:.3f}, BitAcc={row['bit_acc']:.3f})"
                    for row in worst_identify
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## WAVES High-Severity Results",
            "",
            "| Method | Attack | TPR@1%FPR | ID Top-1 | Bit Acc | SSIM |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    high_rows = [
        row
        for row in rows
        if row["suite"] == "waves_core" and abs(row["strength"] - 0.8) < 1e-9
    ]
    for method in methods:
        for row in [item for item in high_rows if item["method"] == method]:
            lines.append(
                f"| {method} | {row['attack']} | {row['tpr_1fpr']:.3f} | "
                f"{row['id_top1']:.3f} | {row['bit_acc']:.3f} | {row['ssim']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- The pilot uses a small sample, so low-FPR estimates are coarse.",
            "- All samples use fixed key 0; this is not a uniform multi-key evaluation.",
            "- The StirMark-compatible track uses published profile values in Python, not the legacy binary.",
            "- The benchmark covers distortions and geometric desynchronization, not WAVES adversarial attacks.",
            "",
        ]
    )
    (args.result_dir / "DETAILED_ANALYSIS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
