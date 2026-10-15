"""Hard validity first; rank only candidates that passed required checks."""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import settings


QUALITY_DIMENSIONS = (
    "subject_fidelity", "atmosphere_fidelity", "motif_fidelity", "audience_fit",
    "composition_quality", "visual_structure_quality", "direction_distinctiveness",
    "quiet_region_usability", "palette_quality",
)
# Neutral starting point. Human labels are required before tuning weights.
DEFAULT_QUALITY_WEIGHTS = {key: float(settings.quality_weights[key]) for key in QUALITY_DIMENSIONS}


def validator_availability(record: dict) -> dict:
    validation = record.get("validation") or {}
    fidelity = record.get("brief_fidelity") or record.get("subject_adherence") or {}
    ocr = ((validation.get("text") or {}).get("ocr_diagnostics") or {}).get("decision")
    result = {"ocr": ("PASS" if ocr == "complete" else "PARTIAL" if ocr in {"OCR_PARTIAL", "OCR_PARTIAL_SUFFICIENT"}
                      else "UNAVAILABLE" if ocr in {"OCR_UNAVAILABLE", None} else "FAIL"),
              "palette": "PASS" if validation.get("decision") == "accepted" and validation.get("palette") else
                         "FAIL" if validation.get("failure_category") in {"PALETTE_OFF", "PALETTE_ROLE_DRIFT"} else "UNAVAILABLE",
              "human": "FAIL" if (validation.get("human") or {}).get("detected") else
                       "PASS" if validation.get("human") else "UNAVAILABLE"}
    available = fidelity.get("dimension_availability") or {}
    for key in ("subject_score", "atmosphere_score", "motif_score", "audience_fit_score"):
        result[key] = available.get(key, "UNAVAILABLE")
    return result


@dataclass(frozen=True, slots=True)
class CandidateQualityScore:
    dimensions: dict[str, float | None]
    availability: dict[str, str]
    overall_quality: float | None
    weights: dict[str, float]

    def record(self) -> dict:
        return {"dimensions": self.dimensions, "availability": self.availability,
                "overall_quality": self.overall_quality, "weights": self.weights}


def hard_validity(record: dict, *, require_vision: bool = False, require_ocr: bool = False) -> dict:
    failures = []
    validation = record.get("validation") or {}
    if validation.get("decision") == "rejected" or record.get("status") in {"rejected", "failed", "runtime_failed"}:
        failures.append(validation.get("failure_category") or record.get("primary_failure_category") or "VALIDATION_FAILED")
    if validation.get("decision") not in {"accepted", "rejected"}:
        failures.append("VALIDATION_UNAVAILABLE")
    fidelity = record.get("brief_fidelity") or record.get("subject_adherence") or {}
    if fidelity.get("decision") in {"SUBJECT_DRIFT", "BRIEF_FIDELITY_DRIFT"}:
        failures.append(fidelity.get("failure_category") or fidelity["decision"])
    thresholds = fidelity.get("thresholds") or {}
    for dimension in ("subject_score", "atmosphere_score", "motif_score", "audience_fit_score"):
        score = fidelity.get(dimension)
        limit = thresholds.get(dimension, settings.subject_adherence_threshold if dimension == "subject_score"
                               else settings.brief_fidelity_threshold)
        if isinstance(score, (float, int)) and score < limit:
            failures.append(dimension.upper() + "_BELOW_MINIMUM")
    if require_vision and fidelity.get("decision") != "pass":
        failures.append("VISION_UNAVAILABLE" if fidelity.get("decision") == "unavailable" else "VISION_REQUIRED")
    ocr = (validation.get("text") or {}).get("ocr_diagnostics") or {}
    if require_ocr and ocr.get("decision") not in {"complete", "OCR_PARTIAL_SUFFICIENT"}:
        failures.append("OCR_UNAVAILABLE" if not ocr else ocr.get("decision") or "OCR_UNAVAILABLE")
    if record.get("direction_similarity_score", 0) >= .88:
        failures.append("DIRECTION_DUPLICATE")
    return {"passed": not failures, "failures": sorted(set(failures))}


def score_candidate(record: dict, *, weights: dict[str, float] | None = None,
                    require_vision: bool = False, require_ocr: bool = False) -> dict:
    gate = hard_validity(record, require_vision=require_vision, require_ocr=require_ocr)
    if not gate["passed"]:
        return {"hard_gate": gate, "quality": None,
                "validator_availability": validator_availability(record)}
    weights = dict(DEFAULT_QUALITY_WEIGHTS if weights is None else weights)
    if set(weights) != set(QUALITY_DIMENSIONS) or any(value < 0 for value in weights.values()):
        raise ValueError("Quality weights must cover every dimension with nonnegative values")
    fidelity = record.get("brief_fidelity") or record.get("subject_adherence") or {}
    validation = record.get("validation") or {}
    quality = validation.get("quality") or {}
    palette = validation.get("palette") or {}
    raw = {
        "subject_fidelity": fidelity.get("subject_score"),
        "atmosphere_fidelity": fidelity.get("atmosphere_score"),
        "motif_fidelity": fidelity.get("motif_score"),
        "audience_fit": fidelity.get("audience_fit_score"),
        "composition_quality": fidelity.get("background_usability_score", quality.get("composition_score")),
        "visual_structure_quality": quality.get("macro_structure_score"),
        "direction_distinctiveness": (1 - record["direction_similarity_score"]
                                      if isinstance(record.get("direction_similarity_score"), (float, int)) else None),
        "quiet_region_usability": quality.get("quiet_region_score"),
        "palette_quality": palette.get("quality_score"),
    }
    dimensions = {key: float(value) if isinstance(value, (float, int)) and 0 <= value <= 1 else None
                  for key, value in raw.items()}
    available = {key: value for key, value in dimensions.items() if value is not None and weights[key] > 0}
    total = sum(weights[key] for key in available)
    overall = sum(value * weights[key] for key, value in available.items()) / total if total else None
    result = CandidateQualityScore(dimensions, {key: "PASS" if value is not None else "UNAVAILABLE"
                                                  for key, value in dimensions.items()}, overall, weights)
    return {"hard_gate": gate, "quality": result.record(),
            "validator_availability": validator_availability(record)}


def rank_candidates(records: list[dict], **options) -> dict:
    evaluations = [(record, score_candidate(record, **options)) for record in records]
    valid = [(record, result["quality"]) for record, result in evaluations
             if result["quality"] is not None and result["quality"]["overall_quality"] is not None]
    # Higher score wins; exact ties use ascending reference for stable selection.
    valid.sort(key=lambda pair: (-pair[1]["overall_quality"],
                                 str(pair[0].get("reference_number") or "")))
    if not valid:
        return {"chosen_reference": None, "compared_candidates": len(records),
                "hard_gate_passed": any(result["hard_gate"]["passed"] for _, result in evaluations),
                "selection_reason": "No hard-valid candidate has usable soft quality signals" if any(
                    result["hard_gate"]["passed"] for _, result in evaluations) else
                    "No candidate passed hard validity gates"}
    chosen, quality = valid[0]
    scores = quality["dimensions"]
    ordered = sorted(((name, value) for name, value in scores.items() if value is not None), key=lambda pair: pair[1])
    return {"chosen_reference": chosen.get("reference_number"),
            "compared_candidates": len(records), "hard_gate_passed": True,
            "quality_score": quality["overall_quality"],
            "strongest_dimensions": [key for key, _ in ordered[-3:]],
            "weaker_dimensions": [key for key, _ in ordered[:3]],
            "selection_reason": "Highest available weighted quality; ties use ascending reference"}
