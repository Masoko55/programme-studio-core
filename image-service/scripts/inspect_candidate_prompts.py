"""Print persisted candidate prompts, sampling settings and QA diagnostics.

The script understands both normal production candidate records and the
probe-only diagnostics created by probe_native_candidate.py.
"""

from __future__ import annotations

import argparse
import json
import re

from pathlib import Path
from typing import Any


# ============================================================
# Helpers
# ============================================================


def _off_palette_ratio(
    item: dict,
) -> float | None:
    value = (
        item.get(
            "off_palette_ratio"
        )
    )

    if (
        value
        is not None
    ):
        try:
            return float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    reason = (
        item.get(
            "rejection_reason"
        )
        or ""
    )

    match = re.search(
        r"off-palette ratio ([0-9.]+)",
        reason,
    )

    if not match:
        return None

    return float(
        match.group(
            1
        )
    )


def _raw_value(
    item: dict,
    key: str,
) -> Any:
    """Prefer probe raw QA, then normal raw validation, then top level."""

    probe_raw = (
        item.get(
            "probe_raw_validation"
        )
        or {}
    )

    if (
        key
        in probe_raw
    ):
        return probe_raw[
            key
        ]

    raw = (
        item.get(
            "raw_validation"
        )
        or {}
    )

    if (
        key
        in raw
    ):
        return raw[
            key
        ]

    return item.get(
        key
    )


def _attempts(
    record: dict,
) -> list[dict]:
    attempts: list[dict] = []

    attempts.extend(
        record.get(
            "rejected_attempts",
            [],
        )
        or []
    )

    attempts.extend(
        record.get(
            "failed_attempts",
            [],
        )
        or []
    )

    attempts.append(
        record
    )

    return sorted(
        attempts,
        key=lambda value: int(
            value.get(
                "attempt_count"
            )
            or 0
        ),
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--reference",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/data/programmes"
        ),
    )

    args = parser.parse_args()

    root = (
        args.data_root
        / args.reference
        / "backgrounds"
    )

    for engine in (
        "flux-2",
        "sdxl-1-0",
        "sd-3-5-medium",
    ):
        for direction in (
            "a",
            "b",
            "c",
        ):
            path = (
                root
                / engine
                / (
                    "image-"
                    + direction
                    + ".json"
                )
            )

            if not path.exists():
                continue

            record = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            for item in _attempts(
                record
            ):
                prompt = (
                    item.get(
                        "compiled_prompt"
                    )
                    or {}
                )

                profile = (
                    item.get(
                        "sampling_profile"
                    )
                    or {}
                )

                output = {
                    "engine": (
                        engine
                    ),
                    "direction": (
                        direction.upper()
                    ),
                    "direction_role": (
                        item.get(
                            "direction_role"
                        )
                    ),
                    "attempt": (
                        item.get(
                            "attempt_count"
                        )
                    ),
                    "status": (
                        item.get(
                            "status"
                        )
                        or "rejected"
                    ),
                    "rejection_reason": (
                        item.get(
                            "rejection_reason"
                        )
                    ),
                    "outcome_category": (
                        item.get(
                            "outcome_category"
                        )
                    ),
                    "failure_category": (
                        item.get(
                            "failure_category"
                        )
                    ),
                    "retry_stage": (
                        item.get(
                            "retry_stage"
                        )
                    ),
                    "spec_sha256": (
                        item.get(
                            "spec_sha256"
                        )
                    ),
                    "seed": (
                        item.get(
                            "seed"
                        )
                    ),

                    # ==========================================
                    # Sampling
                    # ==========================================

                    "sampling_profile": (
                        profile
                    ),
                    "sampler": (
                        profile.get(
                            "sampler_name"
                        )
                    ),
                    "scheduler": (
                        profile.get(
                            "scheduler"
                        )
                    ),
                    "steps": (
                        profile.get(
                            "steps"
                        )
                    ),
                    "cfg": (
                        profile.get(
                            "cfg"
                        )
                    ),
                    "shift": (
                        profile.get(
                            "shift"
                        )
                    ),
                    "shift_mode": (
                        profile.get(
                            "shift_mode"
                        )
                    ),

                    # ==========================================
                    # Probe image preservation
                    # ==========================================

                    "probe_mode": (
                        item.get(
                            "probe_mode"
                        )
                    ),
                    "rejected_image_preserved": (
                        item.get(
                            "rejected_image_preserved"
                        )
                    ),
                    "rejected_image_path": (
                        item.get(
                            "rejected_image_path"
                        )
                    ),
                    "probe_raw_validation_error": (
                        item.get(
                            "probe_raw_validation_error"
                        )
                    ),
                    "probe_full_validation_error": (
                        item.get(
                            "probe_full_validation_error"
                        )
                    ),

                    # ==========================================
                    # Palette
                    # ==========================================

                    "off_palette_ratio": (
                        _off_palette_ratio(
                            item
                        )
                    ),
                    "raw_palette_error": (
                        item.get(
                            "raw_palette_error"
                        )
                    ),
                    "palette_normalized": (
                        item.get(
                            "palette_normalized"
                        )
                    ),

                    # ==========================================
                    # Structural QA
                    # ==========================================

                    "fine_horizontal_pair_ratio": (
                        _raw_value(
                            item,
                            "fine_horizontal_pair_ratio",
                        )
                    ),
                    "fine_vertical_pair_ratio": (
                        _raw_value(
                            item,
                            "fine_vertical_pair_ratio",
                        )
                    ),
                    "structural_edge_density": (
                        _raw_value(
                            item,
                            "structural_edge_density",
                        )
                    ),
                    "raw_edge_density": (
                        _raw_value(
                            item,
                            "raw_edge_density",
                        )
                    ),
                    "outer_edge_density": (
                        _raw_value(
                            item,
                            "outer_edge_density",
                        )
                    ),
                    "center_edge_density": (
                        _raw_value(
                            item,
                            "center_edge_density",
                        )
                    ),
                    "coherent_edge_ratio": (
                        _raw_value(
                            item,
                            "coherent_edge_ratio",
                        )
                    ),
                    "outer_structure_ratio": (
                        _raw_value(
                            item,
                            "outer_structure_ratio",
                        )
                    ),
                    "safe_region_edge_density": (
                        _raw_value(
                            item,
                            "safe_region_edge_density",
                        )
                    ),
                    "entropy": (
                        _raw_value(
                            item,
                            "entropy",
                        )
                    ),
                    "colour_entropy": (
                        _raw_value(
                            item,
                            "colour_entropy",
                        )
                    ),

                    # ==========================================
                    # OCR / human diagnostics
                    # ==========================================

                    "person_detection_count": (
                        _raw_value(
                            item,
                            "person_detection_count",
                        )
                    ),
                    "face_detection_count": (
                        _raw_value(
                            item,
                            "face_detection_count",
                        )
                    ),
                    "profile_detection_count": (
                        _raw_value(
                            item,
                            "profile_detection_count",
                        )
                    ),
                    "upper_body_detection_count": (
                        _raw_value(
                            item,
                            "upper_body_detection_count",
                        )
                    ),

                    # ==========================================
                    # Prompts
                    # ==========================================

                    "positive": (
                        prompt.get(
                            "positive"
                        )
                    ),
                    "negative": (
                        prompt.get(
                            "negative"
                        )
                    ),
                    "clip_l": (
                        prompt.get(
                            "clip_l"
                        )
                    ),
                    "clip_g": (
                        prompt.get(
                            "clip_g"
                        )
                    ),
                    "t5": (
                        prompt.get(
                            "t5"
                        )
                    ),
                }

                print(
                    json.dumps(
                        output,
                        ensure_ascii=False,
                    )
                )


if __name__ == "__main__":
    main()