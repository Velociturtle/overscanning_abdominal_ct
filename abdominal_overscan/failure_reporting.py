"""Small CSV reports for cases that a pipeline stage could not measure."""
from __future__ import annotations

import csv
from pathlib import Path


def write_failed_cases(csv_path: Path, stage: str, failures: list[dict[str, str]]) -> Path:
    """Write or clear the failed-case report for one pipeline stage."""
    report_path = Path(csv_path).with_name(f"{stage}_failed_cases.csv")
    if not failures:
        try:
            report_path.unlink()
        except FileNotFoundError:
            pass
        return report_path

    rows = [
        {
            "stage": stage,
            "file_name": str(item.get("file_name", "")),
            "reason": str(item.get("reason", "")),
        }
        for item in failures
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "file_name", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    return report_path