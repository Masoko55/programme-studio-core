"""Optional image-text subject check through the existing Ollama service.

The checker is enabled by SUBJECT_VISION_MODEL. An unavailable model produces
an explicit unchecked diagnostic; it never masquerades as a passing score.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path

import httpx
from PIL import Image

from app.config.settings import settings


SEMANTIC_FAILURE_CATEGORIES = {
    "subject_score": "SUBJECT_DRIFT",
    "atmosphere_score": "ATMOSPHERE_DRIFT",
    "motif_score": "MOTIF_DRIFT",
    "treatment_score": "TREATMENT_DRIFT",
    "audience_fit_score": "AUDIENCE_DRIFT",
    "unrequested_literal_object_visible": "UNREQUESTED_LITERAL_OBJECT",
    "unrequested_text_visible": "TEXT_OR_LOGO",
}


def select_vision_endpoint(model_name: str) -> str:
    """Use the endpoint verified for the installed MiniCPM vision model."""
    return "/api/chat" if model_name.strip().casefold().startswith("minicpm-v4.5") else "/api/generate"


def _model_answer(payload: dict, endpoint: str) -> dict:
    """Decode the Ollama envelope and one complete JSON object in its text."""
    message = payload.get("message")

    if endpoint == "/api/chat" and not isinstance(message, dict):
        raise ValueError("Missing Ollama chat message")

    content = (
        message.get("content")
        if endpoint == "/api/chat"
        else payload.get("response")
    )

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty or missing model response")

    content = content.strip()
    if content.startswith("```"):
        content = content[3:]
        if content[:4].casefold() == "json":
            content = content[4:]
        content = content.lstrip()
    if content.endswith("```"):
        content = content[:-3].rstrip()

    try:
        answer = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")

        if start < 0:
            raise

        answer, _ = json.JSONDecoder().raw_decode(
            content[start:]
        )

    if isinstance(answer, str):
        answer = json.loads(answer)

    if not isinstance(answer, dict):
        raise ValueError("Model response must be an object")

    return answer


def brief_fidelity_record(result: dict) -> dict:
    """Persist the bounded event-fit contract without the raw model response."""
    return {
        key: result.get(key)
        for key in (
            "checked",
            "subject_score",
            "atmosphere_score",
            "motif_score",
            "treatment_score",
            "audience_fit_score",
            "thresholds",
            "decision",
            "reason",
            "observed_evidence",
            "model",
            "failure_category",
            "dimension_availability",
            "failed_dimensions",
            "background_usability_score",
            "vision_human_signal",
            "vision_debug",
            "hard_dimensions",
            "hard_failed_dimensions",
            "compound_fidelity_failure",
            "human_description",
            "vision_text_signal",
            "vision_text_evidence",
            "unrequested_literal_object_visible",
        )
    }


def hard_fidelity_failure(result: dict | None) -> bool:
    """The single semantic acceptance rule for raw and recovered images."""
    return bool(
        result
        and (
            result.get("hard_failed_dimensions")
            or result.get("decision") == "SUBJECT_DRIFT"
            or (result.get("decision") == "unavailable" and result.get("thresholds"))
        )
    )


def _extract_numeric_score(
    value,
    dimension: str,
    diagnostics: list[dict],
) -> float:
    """Parse a bounded MiniCPM score envelope without guessing among fields."""
    entry = {
        "dimension_name": dimension,
        "input_type": type(value).__name__,
        "extracted_field": None,
        "parsed_value": None,
        "parse_failure_reason": None,
    }

    try:
        if isinstance(value, dict):
            keys = [
                key
                for key in (
                    dimension,
                    "score",
                    "value",
                    "rating",
                )
                if key in value
            ]

            if len(keys) != 1:
                raise ValueError(
                    "ambiguous or missing score field"
                )

            entry["extracted_field"] = keys[0]
            value = value[
                keys[0]
            ]

            if isinstance(value, dict):
                inner = [
                    key
                    for key in (
                        "score",
                        "value",
                        "rating",
                    )
                    if key in value
                ]

                if len(inner) != 1:
                    raise ValueError(
                        "ambiguous nested score"
                    )

                entry[
                    "extracted_field"
                ] += (
                    "."
                    + inner[0]
                )

                value = value[
                    inner[0]
                ]

        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                (
                    int,
                    float,
                    str,
                ),
            )
        ):
            raise ValueError(
                "score is not a numeric scalar"
            )

        score = float(
            value
        )

        if (
            not math.isfinite(
                score
            )
            or not 0 <= score <= 1
        ):
            raise ValueError(
                "score must be finite and between zero and one"
            )

        entry[
            "parsed_value"
        ] = score

        return score

    except (
        TypeError,
        ValueError,
    ) as error:
        entry[
            "parse_failure_reason"
        ] = str(
            error
        )[:100]

        raise ValueError(
            f"{dimension}: "
            f"{entry['parse_failure_reason']}"
        ) from error

    finally:
        diagnostics.append(
            entry
        )


def assess_subject_adherence(
    path: Path,
    visual_design: dict,
    *,
    model: str | None = None,
) -> dict:
    subject = str(
        visual_design.get(
            "primary_visual_subject"
        )
        or ""
    ).strip()

    anchors = [
        str(
            value
        )
        for value
        in visual_design.get(
            "visual_anchors",
            [],
        )
        if str(
            value
        ).strip()
    ]

    chosen_model = (
        settings.subject_vision_model
        if model is None
        else model
    )

    threshold = (
        settings.subject_adherence_threshold
    )

    fidelity_threshold = (
        settings.brief_fidelity_threshold
    )

    semantics = (
        visual_design.get(
            "visual_semantics"
        )
        or {}
    )

    motifs = [
        str(
            value
        )
        for value
        in semantics.get(
            "motif_descriptors",
            [],
        )
        if str(
            value
        ).strip()
    ]

    motifs += [
        str(
            value
        )
        for value
        in visual_design.get(
            "symbolic_motifs",
            [],
        )
        if str(
            value
        ).strip()
    ]

    atmosphere = (
        ", ".join(
            semantics.get(
                "atmosphere_descriptors",
                [],
            )
        )
        or str(
            visual_design.get(
                "creative_description"
            )
            or visual_design.get(
                "mood"
            )
            or ""
        ).strip()
    )

    audience = (
        ", ".join(
            semantics.get(
                "audience_descriptors",
                [],
            )
        )
        or str(
            visual_design.get(
                "audience_context"
            )
            or visual_design.get(
                "age_group"
            )
            or ""
        ).strip()
    )

    motif_structure = str(
        semantics.get(
            "motif_structure_description"
        )
        or ""
    ).strip()

    prominence = (
        visual_design.get(
            "visual_hierarchy"
        )
        or {}
    ).get(
        "motif_prominence",
        "supporting",
    )

    incompatible = ", ".join(
        semantics.get(
            "incompatible_treatments",
            [],
        )
    )

    event = str(
        visual_design.get(
            "event_context"
        )
        or ""
    ).strip()

    treatment = str(
        visual_design.get(
            "theme_reference_treatment"
        )
        or ""
    ).strip()

    required = {
        "subject_score": threshold,
    }

    if atmosphere or event:
        required[
            "atmosphere_score"
        ] = fidelity_threshold

    if (
        motifs
        or motif_structure
        or treatment
    ):
        required[
            "motif_score"
        ] = fidelity_threshold

    if audience:
        required[
            "audience_fit_score"
        ] = fidelity_threshold

    provenance = visual_design.get("semantic_provenance") or {}
    explicit_treatment = (provenance.get("theme_reference_treatment") in
                          {"USER_EXPLICIT", "GRILL_ME_CLARIFIED"})
    if (treatment and explicit_treatment and re.search(
            r"\b(?:viewpoint|perspective|traversal|movement|motion|aerial|first[- ]person)\b",
            treatment, re.I)):
        required["treatment_score"] = fidelity_threshold

    base = {
        "checked": False,
        "score": None,
        "threshold": threshold,
        "subject": subject,
        "anchors": anchors,
        "model": chosen_model,
        "decision": "unavailable",
        "reason": "",
        "subject_score": None,
        "atmosphere_score": None,
        "motif_score": None,
        "treatment_score": None,
        "audience_fit_score": None,
        "thresholds": required,
        "observed_evidence": [],
        "background_usability_score": None,
        "vision_human_signal": None,
        "vision_text_signal": None,
        "vision_text_evidence": "",
        "unrequested_literal_object_visible": None,
    }

    if not chosen_model or not subject:
        return {
            **base,
            "reason": (
                "Vision model or requested visual subject "
                "is not configured"
            ),
            "dimension_availability": {
                field: "UNAVAILABLE"
                for field
                in required
            },
        }

    with Image.open(
        path
    ) as source:
        image = source.convert(
            "RGB"
        )

        image.thumbnail(
            (
                768,
                768,
            ),
            Image.Resampling.LANCZOS,
        )

        buffer = (
            io.BytesIO()
        )

        image.save(
            buffer,
            format="JPEG",
            quality=85,
        )

    prompt = (
        "Assess this background against the requested brief using independent scores between 0 and 1: "
        "subject_score, atmosphere_score, motif_score, audience_fit_score, and treatment_score when requested. "
        "Judge visible subject, global atmosphere, motif topology, and audience fit separately. "
        "Also give background_usability_score for hierarchy, controlled density, and usable calm space, "
        "and human_visible as a boolean based on visible human evidence, including small or distant people. "
        "Give human_description with concrete visible body, face, posture and location evidence when a person is present. "
        "Do not judge colours. Do not infer a subject or motif from decorative lines alone. "
        "A motif pattern does not authorize an unrequested literal source object; count such an object against subject fit. "
        "Intentional abstract or sculptural subjects are valid when requested. "
        "Compare visible connected/repeated motif geometry with the supplied motif topology; "
        "Score treatment_score independently for visible viewpoint, motion and spatial relationship; "
        "do not score likeness to a named reference or require its character. "
        "theme resemblance alone is insufficient. "
        "A subtle motif need not dominate, while a primary motif should be clearly visible. "
        "Assess whether subject presence has become full-frame subject saturation or concept-art clutter. "
        "Reject visible treatments incompatible with the requested mood. "
        "Return one JSON object with numeric subject_score, atmosphere_score, motif_score, "
        "audience_fit_score, treatment_score, background_usability_score (0 to 1); "
        "boolean human_visible and unrequested_literal_object_visible; "
        "boolean unrequested_text_visible and text_evidence describing any visible words, letters, "
        "numbers, logos, signage or grouped pseudo-glyphs and their location; "
        "observed_evidence as a list of visible forms; and a short reason. "
        "Always include unrequested_literal_object_visible as a JSON boolean, even when false. "
        "Requested subject: "
        + subject
        + ". Anchors: "
        + "; ".join(
            anchors[:5]
        )
        + ". Event: "
        + event
        + ". Atmosphere: "
        + atmosphere[:350]
        + ". Audience: "
        + audience
        + ". Theme treatment: "
        + treatment[:200]
        + ". Explicit audience requirement: "
        + str(
            visual_design.get(
                "audience_context"
            )
            or visual_design.get(
                "age_group"
            )
            or ""
        )
        + ". Motifs: "
        + "; ".join(
            motifs[:5]
        )
        + ". Motif structure: "
        + motif_structure[:350]
        + ". Motif literalization policy: "
        + str(visual_design.get("motif_literalization_policy") or "literal_allowed")
        + ". Explicitly authorized literal subject: "
        + str(visual_design.get("authorized_literal_subject") or "none")[:200]
        + ". Set unrequested_literal_object_visible true only for a clearly visible literal object "
        "introduced solely from a motif and absent from the explicitly authorized subject; otherwise set false"
        + ". Motif prominence: "
        + prominence
        + ". Incompatible treatments: "
        + incompatible[:350]
        + "."
    )

    endpoint = (
        select_vision_endpoint(
            chosen_model
        )
    )

    debug = {
        "endpoint": endpoint,
        "model": chosen_model,
        "score_parsing": [],
    }

    vision_human = None
    human_description = ""

    try:
        encoded_image = (
            base64.b64encode(
                buffer.getvalue()
            ).decode(
                "ascii"
            )
        )

        request = {
            "model": chosen_model,
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": 0,
        }

        if endpoint == "/api/chat":
            request[
                "messages"
            ] = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        encoded_image
                    ],
                }
            ]

        else:
            request.update(
                prompt=prompt,
                images=[
                    encoded_image
                ],
            )

        for assessment_attempt in range(2):
            response = httpx.post(
                settings.ollama_base_url.rstrip("/") + endpoint,
                json=request,
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Ollama response must be an object")
            message = payload.get("message") if endpoint == "/api/chat" else None
            raw_text = message.get("content") if isinstance(message, dict) else payload.get("response")
            thinking = message.get("thinking") if isinstance(message, dict) else payload.get("thinking")
            debug["thinking_present"] = bool(thinking)
            debug["thinking_preview_length"] = len(thinking) if isinstance(thinking, str) else 0
            debug["response_length"] = len(raw_text) if isinstance(raw_text, str) else 0
            debug["final_source"] = ("message.content" if endpoint == "/api/chat" else "response") if isinstance(raw_text, str) and raw_text.strip() else None
            debug["raw_text_preview"] = "<omitted>" if isinstance(raw_text, str) else None
            debug["assessment_attempts"] = assessment_attempt + 1
            try:
                answer = _model_answer(payload, endpoint)
                vision_human = answer.get("human_visible") if isinstance(answer.get("human_visible"), bool) else None
                human_description = str(answer.get("human_description") or "")[:240]
                missing = [field for field in required if field not in answer and
                           not (field == "subject_score" and "score" in answer)]
                if (visual_design.get("motif_literalization_policy") == "motif_only" and
                        not isinstance(answer.get("unrequested_literal_object_visible"), bool)):
                    missing.append("boolean unrequested_literal_object_visible")
                if missing:
                    raise ValueError("Vision response lacks " + ", ".join(missing))
                for field in required:
                    _extract_numeric_score(answer.get(field, answer.get("score") if field == "subject_score" else None),
                                           field, [])
                if not isinstance(answer.get("observed_evidence", answer.get("matched_anchors", [])), list):
                    raise ValueError("Vision response lacks observed evidence")
                break
            except (ValueError, TypeError):
                if assessment_attempt:
                    raise

        debug[
            "http_status"
        ] = response.status_code

        debug[
            "response_content_type"
        ] = response.headers.get(
            "content-type",
            "",
        )[:80]

        debug[
            "response_shape"
        ] = (
            sorted(
                str(
                    key
                )
                for key
                in payload
            )[:20]
            if isinstance(
                payload,
                dict,
            )
            else type(
                payload
            ).__name__
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Ollama response must be an object"
            )

        message = (
            payload.get(
                "message"
            )
            if endpoint == "/api/chat"
            else None
        )

        raw_text = (
            message.get(
                "content"
            )
            if isinstance(
                message,
                dict,
            )
            else payload.get(
                "response"
            )
        )

        thinking = (
            message.get(
                "thinking"
            )
            if isinstance(
                message,
                dict,
            )
            else payload.get(
                "thinking"
            )
        )

        debug[
            "thinking_present"
        ] = bool(
            thinking
        )

        debug[
            "thinking_preview_length"
        ] = (
            len(
                thinking
            )
            if isinstance(
                thinking,
                str,
            )
            else 0
        )

        debug[
            "response_length"
        ] = (
            len(
                raw_text
            )
            if isinstance(
                raw_text,
                str,
            )
            else 0
        )

        debug[
            "final_source"
        ] = (
            (
                "message.content"
                if endpoint == "/api/chat"
                else "response"
            )
            if (
                isinstance(
                    raw_text,
                    str,
                )
                and raw_text.strip()
            )
            else None
        )

        if isinstance(
            raw_text,
            str,
        ):
            debug[
                "raw_text_preview"
            ] = "<omitted>"

        answer = (
            _model_answer(
                payload,
                endpoint,
            )
        )

        if (
            visual_design.get("motif_literalization_policy") == "motif_only"
            and not isinstance(answer.get("unrequested_literal_object_visible"), bool)
        ):
            raise ValueError(
                "Vision response lacks boolean unrequested_literal_object_visible"
            )

        vision_human = (
            answer.get(
                "human_visible"
            )
        )

        if not isinstance(
            vision_human,
            bool,
        ):
            vision_human = None

        human_description = str(
            answer.get(
                "human_description"
            )
            or ""
        )[:240]

        scores = {}

        for field in (
            "subject_score",
            "atmosphere_score",
            "motif_score",
            "treatment_score",
            "audience_fit_score",
        ):
            raw_score = answer.get(
                field,
                (
                    answer.get(
                        "score"
                    )
                    if field == "subject_score"
                    else None
                ),
            )

            if raw_score is None:
                if field in required:
                    raise ValueError(
                        f"Vision response lacks {field}"
                    )

                scores[
                    field
                ] = None

                continue

            scores[
                field
            ] = _extract_numeric_score(
                raw_score,
                field,
                debug[
                    "score_parsing"
                ],
            )

        evidence = answer.get(
            "observed_evidence",
            answer.get(
                "matched_anchors",
                [],
            ),
        )

        if not isinstance(
            evidence,
            list,
        ):
            raise ValueError(
                "Vision response lacks observed evidence"
            )

        usability = answer.get(
            "background_usability_score"
        )

        if usability is not None:
            usability = (
                _extract_numeric_score(
                    usability,
                    "background_usability_score",
                    debug[
                        "score_parsing"
                    ],
                )
            )

    except (
        httpx.HTTPError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        debug[
            "parse_error"
        ] = (
            f"{type(error).__name__}: "
            f"{str(error)[:160]}"
        )

        return {
            **base,
            "reason": (
                "Vision check unavailable: "
                f"{type(error).__name__}"
            ),
            "vision_human_signal": vision_human,
            "human_description": human_description,
            "vision_debug": debug,
            "dimension_availability": {
                field: "UNAVAILABLE"
                for field
                in required
            },
        }

    unauthorized_literal = (visual_design.get("motif_literalization_policy") == "motif_only"
                            and answer.get("unrequested_literal_object_visible") is True)
    if unauthorized_literal:
        debug["unrequested_literal_object_visible"] = True

    failed = [
        field
        for field, limit
        in required.items()
        if scores[
            field
        ]
        < limit
    ]

    provenance = (
        visual_design.get(
            "semantic_provenance"
        )
        or {}
    )

    explicit_sources = {
        "USER_EXPLICIT",
        "GRILL_ME_CLARIFIED",
    }

    defining_treatment = bool(
        (
            treatment
            and provenance.get(
                "theme_reference_treatment",
                "USER_EXPLICIT",
            )
            in explicit_sources
        )
        or (
            motif_structure
            and provenance.get(
                "motif_structure_description",
                "USER_EXPLICIT",
            )
            in explicit_sources
        )
        or (
            motifs
            and provenance.get(
                "motif_descriptors",
                "USER_EXPLICIT",
            )
            in explicit_sources
        )
    )

    compound_failure = (
        defining_treatment
        and scores.get(
            "motif_score"
        )
        is not None
        and scores[
            "motif_score"
        ]
        < fidelity_threshold
    )

    hard_dimensions = [
        "subject_score"
    ]

    if defining_treatment:
        hard_dimensions.append(
            "motif_score"
        )
    if "treatment_score" in required:
        hard_dimensions.append("treatment_score")

    if (
        audience
        and (
            provenance.get(
                "audience_descriptors"
            )
            in explicit_sources
            or visual_design.get(
                "audience_context"
            )
            or visual_design.get(
                "age_group"
            )
        )
    ):
        hard_dimensions.append(
            "audience_fit_score"
        )

    if (
        atmosphere
        and "atmosphere_score"
        in visual_design.get(
            "hard_dimensions",
            [],
        )
    ):
        hard_dimensions.append(
            "atmosphere_score"
        )

    hard_failed = [
        field
        for field
        in hard_dimensions
        if field
        in failed
        and (field not in {"subject_score", "treatment_score", "audience_fit_score"}
             or scores[field] < required[field] - 0.10)
    ]

    text_evidence = str(answer.get("text_evidence") or "")[:240].strip()
    text_signal = (answer.get("unrequested_text_visible") is True
                   and bool(re.search(r"\b(?:letters?|words?|numbers?|logos?|signage|glyphs?|"
                                      r"typography|writing|watermark)\b", text_evidence, re.I))
                   and bool(re.search(r"\b(?:upper|lower|left|right|middle|center|centre|"
                                      r"rows?|clusters?|lines?)\b", text_evidence, re.I)))
    if text_signal:
        hard_failed.append("unrequested_text_visible")
    if unauthorized_literal:
        hard_failed.append("unrequested_literal_object_visible")

    # Only a real subject failure should become SUBJECT_DRIFT.
    #
    # A defining explicit motif can still be a hard semantic failure, but it
    # must remain MOTIF_DRIFT so retry compilation strengthens the motif
    # instead of unnecessarily rebuilding the whole requested subject.
    if text_signal:
        decision = "TEXT_OR_LOGO"
    elif unauthorized_literal:
        decision = "UNREQUESTED_LITERAL_OBJECT"
    elif (
        "subject_score"
        in hard_failed
    ):
        decision = (
            "SUBJECT_DRIFT"
        )

    elif "treatment_score" in hard_failed:
        decision = "TREATMENT_DRIFT"

    elif failed:
        decision = (
            "BRIEF_FIDELITY_DRIFT"
        )

    else:
        decision = (
            "pass"
        )

    worst = (
        hard_failed[
            0
        ]
        if hard_failed
        else (
            min(
                failed,
                key=lambda field: (
                    scores[
                        field
                    ]
                    / required[
                        field
                    ]
                ),
            )
            if failed
            else None
        )
    )

    return {
        **base,
        "checked": True,
        "score": scores[
            "subject_score"
        ],
        **scores,
        "decision": decision,
        "failure_category": ("TEXT_OR_LOGO" if text_signal else
                             "UNREQUESTED_LITERAL_OBJECT" if unauthorized_literal else
                             SEMANTIC_FAILURE_CATEGORIES[worst] if worst else None),
        "unrequested_literal_object_visible": unauthorized_literal,
        "vision_text_signal": text_signal,
        "vision_text_evidence": text_evidence if text_signal else "",
        "compound_fidelity_failure": (
            compound_failure
        ),
        "hard_dimensions": (
            hard_dimensions
        ),
        "hard_failed_dimensions": (
            hard_failed
        ),
        "failed_dimensions": (
            failed
        ),
        "background_usability_score": (
            usability
        ),
        "vision_human_signal": (
            vision_human
        ),
        "human_description": (
            human_description
        ),
        "dimension_availability": {
            field: (
                (
                    "FAIL"
                    if field
                    in failed
                    else "PASS"
                )
                if scores.get(
                    field
                )
                is not None
                else "UNAVAILABLE"
            )
            for field
            in required
        },
        "reason": str(
            answer.get(
                "reason"
            )
            or ""
        )[:300],
        "observed_evidence": [
            str(
                item
            )[:120]
            for item
            in evidence[:8]
        ],
        "model": (
            chosen_model
        ),
        "vision_debug": (
            debug
        ),
    }
