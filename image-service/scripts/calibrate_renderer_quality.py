"""Compare candidate diagnostics with human labels; never invent calibration data."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from report_renderer_benchmark import load_records


RATINGS = ("subject", "atmosphere", "motif", "audience_fit", "overall")


def calibrate(records: list[dict], labels: list[dict]) -> dict | str:
    if not labels:
        return "NO HUMAN CALIBRATION DATA"
    by_reference = {str(row.get("candidate_reference") or row.get("reference")): row for row in labels
                    if row.get("candidate_reference") or row.get("reference")}
    pairs = [(record, by_reference.get(str(record.get("reference_number")))) for record in records]
    pairs = [(record, label) for record, label in pairs if label]
    if not pairs:
        return "NO HUMAN CALIBRATION DATA"
    errors = {}
    for name in RATINGS:
        differences = []
        for record, label in pairs:
            human = label.get("human_" + name + "_rating")
            model = ((record.get("brief_fidelity") or {}).get(name + "_score") if name != "overall"
                     else ((record.get("quality_evaluation") or {}).get("quality") or {}).get("overall_quality"))
            if isinstance(human, (int, float)) and 1 <= human <= 5 and isinstance(model, (int, float)):
                differences.append(abs(float(model) - (float(human) - 1) / 4))
        errors[name] = round(statistics.mean(differences), 4) if differences else None
    confusion = {"false_accept": 0, "false_reject": 0, "labelled": 0}
    for record, label in pairs:
        if not isinstance(label.get("acceptable"), bool):
            continue
        confusion["labelled"] += 1
        selected = (record.get("quality_evaluation") or {}).get("hard_gate", {}).get("passed") is True
        confusion["false_accept"] += selected and not label["acceptable"]
        confusion["false_reject"] += not selected and label["acceptable"]
    ranked = [(float(((record.get("quality_evaluation") or {}).get("quality") or {}).get("overall_quality")),
               float(label["human_overall_rating"])) for record, label in pairs
              if isinstance(((record.get("quality_evaluation") or {}).get("quality") or {}).get("overall_quality"), (int, float))
              and isinstance(label.get("human_overall_rating"), (int, float))]
    correlation = None
    if len(ranked) >= 3:
        def ranks(values):
            return [sum(other < value for other in values) + (sum(other == value for other in values) - 1) / 2
                    for value in values]
        left, right = (ranks([pair[index] for pair in ranked]) for index in (0, 1))
        left_mean, right_mean = statistics.mean(left), statistics.mean(right)
        numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
        denominator = (sum((x - left_mean) ** 2 for x in left) *
                       sum((y - right_mean) ** 2 for y in right)) ** .5
        correlation = round(numerator / denominator, 4) if denominator else None
    groups = defaultdict(list)
    for record, label in pairs:
        if isinstance(label.get("preferred"), bool):
            groups[str(label.get("brief_identifier") or record.get("source_reference") or "")].append((record, label))
    comparable = [rows for rows in groups.values() if len(rows) >= 2 and any(label["preferred"] for _, label in rows)]
    matches = 0
    for rows in comparable:
        ranked_rows = [(record, label) for record, label in rows
                       if (record.get("quality_evaluation") or {}).get("hard_gate", {}).get("passed")
                       and isinstance(((record.get("quality_evaluation") or {}).get("quality") or {}).get("overall_quality"), (int, float))]
        if ranked_rows:
            chosen = max(ranked_rows, key=lambda pair: pair[0]["quality_evaluation"]["quality"]["overall_quality"])
            matches += bool(chosen[1]["preferred"])
    agreement = matches / len(comparable) if comparable else None
    return {"human_labelled_pairs": len(pairs), "mean_absolute_error": errors,
            "false_accept_rate": (confusion["false_accept"] / confusion["labelled"] if confusion["labelled"] else None),
            "false_reject_rate": (confusion["false_reject"] / confusion["labelled"] if confusion["labelled"] else None),
            "rank_correlation": correlation, "selection_agreement": agreement,
            "note": "Calibration requires multiple independently human-rated candidates per brief."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--human-labels", type=Path)
    args = parser.parse_args()
    labels = json.loads(args.human_labels.read_text(encoding="utf-8")) if args.human_labels and args.human_labels.exists() else []
    result = calibrate(load_records(args.records), labels)
    print(result if isinstance(result, str) else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
