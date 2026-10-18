"""Export candidate metadata for blind human review without model quality scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_renderer_benchmark import load_records


def blind_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        manifest = record.get("reproducibility_manifest") or {}
        rows.append({"candidate_reference": record.get("reference_number"),
                     "brief_identifier": manifest.get("source_reference") or record.get("reference_number"),
                     "direction": record.get("direction_id"),
                     "image_path": record.get("output_path"),
                     "semantic_contract_summary": {
                         "subject": record.get("visual_anchors"),
                         "palette": record.get("resolved_palette"),
                         "semantic_contract_sha256": record.get("semantic_contract_sha256")},
                     "human_subject_rating": None, "human_atmosphere_rating": None,
                     "human_motif_rating": None, "human_audience_fit_rating": None,
                     "human_composition_quality_rating": None, "human_overall_rating": None,
                     "preferred": None, "acceptable": None, "notes": ""})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(blind_rows(load_records(args.records)), indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(load_records(args.records))} blind review rows")


if __name__ == "__main__":
    main()
