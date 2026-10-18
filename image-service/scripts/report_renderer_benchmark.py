"""Summarize candidate records by engine/profile/direction/failure, never event name."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    records = []
    for file in files:
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict) and value.get("engine_id") and value.get("direction_id"):
            records.append(value)
    return records


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        key = (record.get("engine_id"), (record.get("sampling_profile") or {}).get("name"),
               record.get("direction_id"), record.get("primary_failure_category") or "PASS",
               record.get("renderer_recovery_type") or "baseline")
        groups[key].append(record)

    def mean(values):
        data = [float(value) for value in values if isinstance(value, (int, float))]
        return round(statistics.mean(data), 4) if data else None

    def percentage(values):
        observed = [value for value in values if value is not None]
        return round(100 * sum(observed) / len(observed), 2) if observed else None

    result = []
    for (engine, profile, direction, failure, recovery), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        fidelity = [row.get("brief_fidelity") or {} for row in rows]
        result.append({"engine": engine, "profile": profile, "direction": direction,
                       "failure_category": failure, "attempts": len(rows),
                       "recovery_category": recovery,
                       "valid_pct": round(100 * sum(row.get("status") == "complete" for row in rows) / len(rows), 2),
                       **{name + "_mean": mean(item.get(name + "_score") for item in fidelity)
                          for name in ("subject", "atmosphere", "motif")},
                       "audience_mean": mean(item.get("audience_fit_score") for item in fidelity),
                       "palette_pass_pct": percentage(
                           (row.get("primary_failure_category") not in {"PALETTE_OFF", "PRIMARY_MISSING", "SECONDARY_MISSING"})
                           if (row.get("validation") or {}).get("palette") else None for row in rows),
                       "ocr_pass_or_partial_pct": percentage(
                           (decision in {"complete", "OCR_PARTIAL"}) if decision is not None else None
                           for decision in (((row.get("validation") or {}).get("text") or {}).get("ocr_diagnostics", {}).get("decision")
                                            for row in rows)),
                       "quality_mean": mean(((row.get("quality_evaluation") or {}).get("quality") or {}).get("overall_quality") for row in rows),
                       "selection_win_count": sum(bool(row.get("selected")) for row in rows)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="Candidate JSON file or directory")
    args = parser.parse_args()
    print(json.dumps(summarize(load_records(args.records)), indent=2))


if __name__ == "__main__":
    main()
