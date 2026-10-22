import json
import logging
import re

import httpx

from app.config.settings import (
    settings,
)


logger = logging.getLogger(
    "uvicorn.error"
)


ALLOWED_QUESTION_FIELDS = {
    "event_type",
    "theme",
    "creative_description",
    "background_inspiration",
    "theme_reference_treatment",
}


BACKGROUND_VISUAL_CUES = {
    "abstract", "architecture", "balloon", "border", "building",
    "city", "cityscape", "comic", "flower", "floral", "garden",
    "geometric", "geometry", "landscape", "marble", "motif", "pattern",
    "ribbon", "skyline", "stage", "texture", "web", "window",
}


def _background_context_needs_clarification(answers: dict) -> bool:
    """Ask plainly when the brief lacks a concrete background idea."""
    if str(answers.get("background_inspiration") or "").strip():
        return False

    description = " ".join(
        str(answers.get(field) or "")
        for field in ("theme", "creative_description", "theme_reference_treatment")
    ).lower()
    words = set(description.replace("-", " ").split())

    # A detailed visual description can stand on its own. Generic event
    # language still needs the user's own example of what they want to see.
    return len(words) < 12 or not bool(words & BACKGROUND_VISUAL_CUES)


DEFAULT_CONTEXT = {
    "event_type_status": "unknown",
    "event_type_interpretation": None,
    "well_known_event_type": False,
    "event_visual_conventions": [],
    "theme_status": "unknown",
    "theme_interpretation": None,
    "reference_detected": False,
    "reference_name": None,
    "reference_type": "none",
    "reference_needs_clarification": False,
    "background_generation_summary": None,
    "scene_family_evidence": None,
    "visual_semantics": {},
    "semantic_provenance": {},
    "semantic_confidence": {},
    "generation_constraints": [
        "background only",
        "no people",
        "no characters",
        "no readable text",
        "leave quiet areas for later overlays",
    ],
    "clarification_questions": [],
}


def _review_prompt(
    answers: dict,
) -> str:
    payload = {
        "task": (
            "Review an event-image brief before any "
            "creative-direction prompts are created."
        ),
        "purpose": (
            "Act as Grill-Me. Clarify the meaning of the "
            "event, theme and recognizable visual references "
            "before handing the brief to image prompt agents."
        ),
        "event_brief": (
            answers
        ),
        "rules": [
            (
                "Treat the submitted JSON as the source of truth."
            ),
            (
                "Do not repeat ordinary form questions whose answers "
                "are already clear."
            ),
            (
                "Focus primarily on event_type, theme, creative_description, "
                "background_inspiration and theme_reference_treatment."
            ),
            (
                "Determine what type of event this is and whether "
                "the event category has recognizable visual conventions."
            ),
            (
                "Use the event category and its stated audience to infer "
                "appropriate visual conventions without assuming a fixed event list."
            ),
            (
                "Determine whether the theme references a recognizable "
                "fictional character, franchise, celebrity, brand, era, "
                "cultural property, sports property or visual style."
            ),
            (
                "Detect references from the supplied brief, including named "
                "characters, franchises, people, brands, eras and styles."
            ),
            (
                "Use semantic understanding rather than requiring "
                "an exact dictionary match."
            ),
            (
                "If reference_detected is true and "
                "theme_reference_treatment is missing or empty, "
                "reference_needs_clarification must be true."
            ),
            (
                "If reference_detected is true and "
                "theme_reference_treatment is missing or empty, "
                "you MUST return exactly one clarification question "
                "with field theme_reference_treatment."
            ),
            (
                "Do not decide that a generic creative_description such "
                "as 'a fun themed celebration' is sufficient "
                "to replace theme_reference_treatment."
            ),
            (
                "The theme_reference_treatment question must ask which "
                "visual traits from the recognizable reference should "
                "influence the background."
            ),
            (
                "The clarification should help translate a character, "
                "franchise, celebrity or brand reference into abstract "
                "visual language rather than directly depicting it."
            ),
            (
                "Useful abstract traits include colours, patterns, "
                "textures, architecture, geometry, atmosphere, era cues, "
                "symbolic motifs and environmental styling."
            ),
            (
                "Never instruct downstream image generation to depict "
                "the recognizable person or character."
            ),
            (
                "For a vague recognizable-reference answer, infer only a few "
                "high-confidence abstract visual traits. Empty semantic fields "
                "are correct when the brief does not support them. Do not invent "
                "materials, textures, lighting, composition or confidence values. "
                "Never infer a logo, emblem, insignia, costume, mask, likeness, "
                "character silhouette or branded typography as a render target."
            ),
            (
                "generation_constraints apply to background generation only: "
                "no readable programme/title text, logos, people or characters. "
                "Leave final composition and accessibility text to a later stage."
            ),
            (
                "If event_type is vague, contradictory or not actually "
                "an event type, ask one focused event_type question."
            ),
            (
                "If theme is missing, contradictory or ambiguous, "
                "ask one focused theme question."
            ),
            (
                "If event_type, theme and creative_description conflict, "
                "ask only the smallest question needed to resolve it."
            ),
            (
                "Ask a background-design question only when the supplied theme "
                "and creative description cannot determine a visual subject or "
                "environment. Ask in everyday language what the person would like "
                "people to see in the background, with examples such as a city "
                "skyline, flowers, balloons, a stage or an abstract pattern. "
                "Use the background_inspiration field for that single answer. "
                "Do not ask about timezone, accessibility, programme rows, venue, "
                "date or start time during semantic review."
            ),
            (
                "Return at most three clarification questions."
            ),
            (
                "Produce a concise background_generation_summary for "
                "the next creative-direction agent."
            ),
            (
                "Derive visual_semantics from the supplied meaning, not from an event-name "
                "lookup. Preserve unusual atmosphere words. Describe motif topology in "
                "motif_structure_description and name visually incompatible treatments "
                "without judging palette or requesting people or text."
            ),
        ],
        "output_format": {
            "event_type_status": (
                "clear or needs_clarification"
            ),
            "event_type_interpretation": (
                "concise interpreted event type or null"
            ),
            "well_known_event_type": (
                "boolean"
            ),
            "event_visual_conventions": [
                "short visual convention"
            ],
            "theme_status": (
                "clear or needs_clarification"
            ),
            "theme_interpretation": (
                "concise interpreted theme or null"
            ),
            "reference_detected": (
                "boolean"
            ),
            "reference_name": (
                "recognized reference name or null"
            ),
            "reference_type": (
                "none, character, franchise, celebrity, brand, "
                "event_style, era, cultural_reference or other"
            ),
            "reference_needs_clarification": (
                "boolean"
            ),
            "background_generation_summary": (
                "concise clarified instructions for the next agent"
            ),
            "visual_semantics": {
                "atmosphere_descriptors": ["short supplied mood phrase"],
                "audience_descriptors": ["short audience description"],
                "motif_descriptors": ["visible motif"],
                "motif_structure_description": "how the motif visibly connects or repeats",
                "lighting_descriptors": ["brief-compatible lighting"],
                "material_descriptors": ["brief-compatible material"],
                "texture_descriptors": ["brief-compatible texture"],
                "incompatible_treatments": ["visually incompatible treatment"],
                "scene_family": "organic, architectural-interior, abstract, or general",
                "composition_affinities": ["brief-compatible structural family"],
            },
            "semantic_confidence": {"inferred field name": "optional number from 0 to 1; omit if unknown"},
            "scene_family_evidence": "optional exact excerpt from the user's scene wording; never infer an excerpt",
            "generation_constraints": [
                "constraint"
            ],
            "clarification_questions": [
                {
                    "field": (
                        "event_type, theme, creative_description, background_inspiration "
                        "or theme_reference_treatment"
                    ),
                    "question": (
                        "direct user-facing clarification question"
                    ),
                    "reason": (
                        "why the clarification is required"
                    ),
                }
            ],
        },
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _theme_reference_treatment(
    answers: dict,
) -> str:
    return str(
        answers.get(
            "theme_reference_treatment"
        )
        or ""
    ).strip()


def _valid_clarification_questions(context: dict) -> list[dict]:
    questions = []

    for item in (
        context.get(
            "clarification_questions"
        )
        or []
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        field = str(
            item.get(
                "field"
            )
            or ""
        ).strip()

        question = str(
            item.get(
                "question"
            )
            or ""
        ).strip()

        reason = str(
            item.get(
                "reason"
            )
            or ""
        ).strip()

        if (
            field
            not in ALLOWED_QUESTION_FIELDS
        ):
            continue

        if not question:
            continue

        questions.append(
            {
                "field": field,
                "question": question,
                "reason": (
                    reason
                    or (
                        "The creative intent "
                        "needs clarification."
                    )
                ),
            }
        )

    return questions


_INFERRED_ASSET = re.compile(
    r"\b(?:logos?|emblems?|insignia|trademarks?|brand(?:ed)? symbols?|"
    r"costumes?|masks?|weapons?|signature props?|character(?: silhouettes?)?|likeness(?:es)?|"
    r"branded typography|headshots?)\b", re.I,
)
_FINAL_COMPOSITION = re.compile(
    r"\b(?:readable|legible|write|written|show|display|include|add|place|"
    r"render|print|visible|clear)\b.{0,55}\b(?:text|words?|wording|"
    r"typography|titles?|programmes?|schedules?|labels?|logos?|headshots?|"
    r"people|persons?|characters?)\b|"
    r"\b(?:programme|program|title|schedule|label|wording)\b.{0,55}"
    r"\b(?:readable|legible|visible)\b|"
    r"\b(?:final composition|accessibility text)\b", re.I,
)


def _vague_reference_treatment(treatment: str, reference_name: str) -> bool:
    reference_pattern = r"[\s_-]+".join(re.escape(part) for part in re.split(r"[\s_-]+", reference_name) if part)
    remainder = re.sub(reference_pattern, " ", treatment, flags=re.I) if reference_pattern else treatment
    words = re.findall(r"[a-z]+", remainder.casefold())
    return bool(words) and set(words) <= {
        "a", "an", "the", "inspired", "by", "visual", "traits", "style",
        "vibes", "look", "feel", "theme", "from", "of", "with",
    }


def _safe_inferred_reference_phrase(phrase: str, reference_name: str) -> bool:
    if _INFERRED_ASSET.search(phrase):
        return False
    reference_pattern = r"[\s_-]+".join(re.escape(part) for part in re.split(r"[\s_-]+", reference_name) if part)
    return not (reference_pattern and re.search(reference_pattern, phrase, re.I))


def _explicit_spatial_treatment(treatment: str) -> list[str]:
    """Normalize only viewpoint and movement actually stated by the user."""
    text = treatment.casefold()
    result = []
    if re.search(r"\bfirst[- ]person\b|\b(?:viewer|camera)\s+(?:is\s+)?(?:moving|travelling|traveling)", text):
        result.append("immersive first-person viewpoint")
    if re.search(r"\b(?:view from above|aerial view|overhead view)\b", text):
        result.append("aerial viewpoint")
    if re.search(r"\b(?:swing\w*|fly\w*|mov\w*|travel\w*|travers\w*|glid\w*|rush\w*)\b", text):
        result.append("forward traversal and receding spatial depth")
    if re.search(r"\b(?:swing\w*|arcing|sweeping)\s+(?:through|across|past|between|over)\b", text):
        result.append("arcing spatial motion")
    return result


def _ground_scene_family(family: str, source: str, confidence: float | None,
                         support: str, evidence: str = "") -> str:
    """Do not turn an inferred motif association into an environment."""
    if family == "general" or source in {"USER_EXPLICIT", "GRILL_ME_CLARIFIED"}:
        return family
    if source != "SEMANTIC_INFERENCE":
        return "general"
    # Explicit wording of the environment is sufficient; for inferred
    # wording, a high score alone never substitutes for textual grounding.
    family_terms = [part for part in re.split(r"[^a-z]+", family) if len(part) > 3]
    words = set(re.findall(r"[a-z]+", support.casefold()))
    direct = any(term in words for term in family_terms)
    quoted = bool(confidence is not None and confidence >= .85 and len(evidence.strip()) >= 4
                  and evidence.casefold() in support.casefold())
    return family if direct or quoted else "general"


def grounded_scene_family(brief: dict) -> str:
    """Apply the same scene evidence rule to a persisted or direct brief."""
    context = brief.get("grill_me_context") or {}
    semantics = brief.get("visual_semantics") or context.get("visual_semantics") or {}
    family = str(semantics.get("scene_family") or "general").strip().casefold()
    if family not in {"organic", "architectural-interior", "abstract", "general"}:
        return "general"
    provenance = (brief.get("semantic_contract") or {}).get("semantic_provenance") or (
        context.get("semantic_provenance") or brief.get("semantic_provenance") or {})
    confidence = (brief.get("semantic_contract") or {}).get("semantic_confidence") or (
        context.get("semantic_confidence") or {})
    support = " ".join(str(brief.get(key) or "") for key in
                       ("background_inspiration", "creative_description"))
    return _ground_scene_family(family, provenance.get("scene_family", "USER_EXPLICIT"),
                                confidence.get("scene_family"), support,
                                str(context.get("scene_family_evidence") or ""))


def _safe_background_constraint(phrase: str) -> bool:
    # Negative exclusions remain useful; affirmative final artwork requests do not.
    if (re.match(r"\s*(?:no|without|avoid|exclude|do not)\b", phrase, re.I)
            and not re.search(r"\b(?:include|add|show|display|place|render|print|write)\b", phrase, re.I)):
        return True
    if re.search(r"\b(?:logos?|emblems?|headshots?)\b", phrase, re.I):
        return False
    return not _FINAL_COMPOSITION.search(phrase)


def _safe_context(
    value,
    answers: dict,
) -> dict:
    if not isinstance(value, dict):
        value = {}

    context = dict(DEFAULT_CONTEXT)
    context.update({key: item for key, item in value.items() if key in context})
    raw_semantics = context.get("visual_semantics")
    if not isinstance(raw_semantics, dict):
        raw_semantics = {}
    supplied_semantics = answers.get("visual_semantics")
    if isinstance(supplied_semantics, dict):
        raw_semantics = {**raw_semantics, **supplied_semantics}

    def descriptors(key: str, fallback: str = "") -> list[str]:
        raw = raw_semantics.get(key)
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            raw = []
        result = [str(item).strip()[:180] for item in raw if str(item).strip()]
        return result[:8] or ([fallback[:180]] if fallback else [])

    family = str(raw_semantics.get("scene_family") or "general").strip().casefold()
    if family not in {"organic", "architectural-interior", "abstract", "general"}:
        family = "general"
    proposed_family = family
    supplied_family = isinstance(supplied_semantics, dict) and "scene_family" in supplied_semantics
    raw_confidence = (value.get("semantic_confidence") or {}).get("scene_family")
    confidence = float(raw_confidence) if isinstance(raw_confidence, (float, int)) and not isinstance(raw_confidence, bool) else None
    scene_support = " ".join(str(answers.get(key) or "") for key in
                             ("background_inspiration", "creative_description"))
    family = _ground_scene_family(family, "USER_EXPLICIT" if supplied_family else
                                  "SEMANTIC_INFERENCE" if "scene_family" in raw_semantics else "SAFE_FALLBACK",
                                  confidence, scene_support,
                                  str(context.get("scene_family_evidence") or ""))
    if proposed_family != "general" and family == "general":
        context["background_generation_summary"] = (None if answers.get("background_inspiration") else
            "Use only the supplied visual motifs and treatment; do not invent an environment.")
    context["visual_semantics"] = {
        "atmosphere_descriptors": descriptors(
            "atmosphere_descriptors", str(answers.get("creative_description") or "")),
        "audience_descriptors": descriptors(
            "audience_descriptors", str(answers.get("age_group") or "")),
        "motif_descriptors": descriptors("motif_descriptors"),
        "motif_structure_description": str(raw_semantics.get("motif_structure_description") or "").strip()[:360],
        "lighting_descriptors": descriptors("lighting_descriptors"),
        "material_descriptors": descriptors("material_descriptors"),
        "texture_descriptors": descriptors("texture_descriptors"),
        "incompatible_treatments": descriptors("incompatible_treatments"),
        "scene_family": family,
        "composition_affinities": descriptors("composition_affinities"),
    }
    questions = _valid_clarification_questions(context)

    event = str(answers.get("event_type") or "").strip()
    theme = str(answers.get("theme") or "").strip()
    if event and re.search(r"\b(birthday|party|wedding|shower|graduation|gala|ceremony|funeral|conference|celebration)\b", event, re.I):
        context.update(event_type_status="known", event_type_interpretation=event,
                       well_known_event_type=True)
    if theme and context.get("theme_status") != "known":
        context.update(theme_status="known", theme_interpretation=theme)
    reference_text = " ".join(
        str(answers.get(field) or "")
        for field in (
            "theme",
            "creative_description",
            "background_inspiration",
            "theme_reference_treatment",
        )
    )
    reference = re.search(
        r"\b(?:a|an|the)?\s*([\w-]{4,})[- ]inspired\b",
        reference_text,
        re.I,
    )
    if (
        reference
        and re.search(
            r"\b(?:\w*hero(?:es)?|characters?|franchises?|fictional)\b",
            reference_text,
            re.I,
        )
        and not context.get("reference_detected")
    ):
        context.update(
            reference_detected=True,
            reference_name=reference.group(1).replace("-", " ").title(),
            reference_type="character",
        )
    if context.get("reference_detected"):
        reference_name = str(context.get("reference_name") or "").strip()
        treatment = _theme_reference_treatment(answers)
        vague = _vague_reference_treatment(treatment, reference_name)
        explicit_spatial = _explicit_spatial_treatment(treatment) if not vague else []
        support = " ".join(str(answers.get(key) or "") for key in (
            "creative_description", "background_inspiration", "theme_reference_treatment",
        )).casefold()
        semantics = context["visual_semantics"]
        for key, content in semantics.items():
            if key == "scene_family":
                if vague and not (isinstance(supplied_semantics, dict) and key in supplied_semantics):
                    semantics[key] = "general"
                continue
            explicit = isinstance(supplied_semantics, dict) and key in supplied_semantics
            if isinstance(content, list):
                semantics[key] = [phrase for phrase in content
                                  if _safe_inferred_reference_phrase(phrase, reference_name)
                                  and (explicit or key not in {
                                      "material_descriptors", "texture_descriptors",
                                      "lighting_descriptors", "composition_affinities",
                                  } or (not vague and phrase.casefold() in support))]
            elif content and not _safe_inferred_reference_phrase(content, reference_name):
                semantics[key] = ""
        context["visual_semantics"] = semantics
        if explicit_spatial:
            semantics["composition_affinities"] = list(dict.fromkeys(
                [*semantics["composition_affinities"], *explicit_spatial]))[:8]
    context["semantic_provenance"] = {
        key: ("SAFE_FALLBACK" if not content or (key == "scene_family" and content == "general"
                                    and not (isinstance(supplied_semantics, dict) and key in supplied_semantics)) else
              "USER_EXPLICIT" if isinstance(supplied_semantics, dict) and key in supplied_semantics else
              "SEMANTIC_INFERENCE" if key in raw_semantics else
              "USER_EXPLICIT" if (key == "atmosphere_descriptors" and answers.get("creative_description"))
              or (key == "audience_descriptors" and answers.get("age_group")) else "SAFE_FALLBACK")
        for key, content in context["visual_semantics"].items()
    }
    if context.get("reference_detected") and _explicit_spatial_treatment(_theme_reference_treatment(answers)):
        context["semantic_provenance"]["composition_affinities"] = "USER_EXPLICIT"
    context["semantic_confidence"] = {
        key: float(score) for key, score in (value.get("semantic_confidence") or {}).items()
        if key in context["visual_semantics"] and context["semantic_provenance"][key] == "SEMANTIC_INFERENCE"
        and isinstance(score, (float, int)) and not isinstance(score, bool) and 0 <= score <= 1
    }
    if not context.get("background_generation_summary"):
        inspiration = str(answers.get("background_inspiration") or "").strip()
        treatment = str(answers.get("theme_reference_treatment") or "").strip()
        if inspiration:
            context["background_generation_summary"] = (
                f"Background artwork: {inspiration}. {treatment}. No people, characters or generated text."
            ).strip()
    questions = [item for item in questions if not str(answers.get(item["field"]) or "").strip()]

    reference_detected = bool(
        context.get(
            "reference_detected"
        )
    )

    # When an event already has a named theme, asking the user to restate the
    # theme competes with the one question that improves image composition.
    # Keep the conversation in everyday language and ask what should appear
    # in the background instead.
    if (
        _background_context_needs_clarification(answers)
        and str(answers.get("theme") or "").strip()
    ):
        questions = [
            item
            for item
            in questions
            if item["field"] != "theme"
        ]

    treatment = (
        _theme_reference_treatment(
            answers
        )
    )

    if (
        reference_detected
        and not treatment
    ):
        context[
            "reference_needs_clarification"
        ] = True

        questions = [
            item
            for item
            in questions
            if (
                item[
                    "field"
                ]
                != "theme_reference_treatment"
            )
        ]

        reference_name = (
            context.get(
                "reference_name"
            )
            or answers.get(
                "theme"
            )
            or "the reference"
        )

        questions.insert(
            0,
            {
                "field": (
                    "theme_reference_treatment"
                ),
                "question": (
                    f"Which visual traits from {reference_name} "
                    "should influence the background? "
                    "For example colours, abstract patterns, "
                    "geometry, textures, architecture or atmosphere."
                ),
                "reason": (
                    "The theme uses a recognizable reference. "
                    "Grill-Me needs to know which visual traits "
                    "should be translated into abstract background "
                    "design without depicting the character or subject."
                ),
            },
        )

    elif (
        reference_detected
        and treatment
    ):
        context[
            "reference_needs_clarification"
        ] = False

        questions = [
            item
            for item
            in questions
            if (
                item[
                    "field"
                ]
                != "theme_reference_treatment"
            )
        ]

    background_fields = {"background_inspiration"}
    has_background_question = any(
        item["field"] in background_fields
        for item in questions
    )

    if (
        _background_context_needs_clarification(answers)
        and not has_background_question
    ):
        questions.append(
            {
                "field": "background_inspiration",
                "question": (
                    "What would you like people to see in the background? "
                    "For example, a city skyline, flowers, balloons, a stage, "
                    "a pattern, or something abstract."
                ),
                "reason": (
                    "A clear visual idea helps us create three strong background "
                    "options that all feel close to what you want."
                ),
            }
        )

    context[
        "clarification_questions"
    ] = questions[
        :3
    ]

    constraints = (
        context.get(
            "generation_constraints"
        )
    )

    if not isinstance(
        constraints,
        list,
    ):
        constraints = []
    constraints = [str(item).strip()[:180] for item in constraints
                   if isinstance(item, str) and item.strip()
                   and _safe_background_constraint(item)]

    required_constraints = [
        "background only",
        "no people",
        "no characters",
        "no logos",
        "no readable text",
        "leave quiet areas for later overlays",
    ]

    for constraint in (
        required_constraints
    ):
        if (
            constraint
            not in constraints
        ):
            constraints.append(
                constraint
            )

    context[
        "generation_constraints"
    ] = constraints

    return context


def review_creative_context(
    answers: dict,
) -> dict:
    url = (
        settings.ollama_base_url
        .rstrip(
            "/"
        )
        + "/api/generate"
    )

    request_body = {
        "model": (
            settings.direction_a_model
        ),
        "prompt": (
            _review_prompt(
                answers
            )
        ),
        "stream": False,
        "format": "json",
        "keep_alive": 0,
        "options": {
            "temperature": 0.1,
        },
    }

    timeout = httpx.Timeout(
        connect=(
            settings
            .ollama_connect_timeout_seconds
        ),
        read=(
            settings
            .ollama_read_timeout_seconds
        ),
        write=30,
        pool=10,
    )

    try:
        with httpx.Client(
            timeout=timeout
        ) as client:
            response = (
                client.post(
                    url,
                    json=request_body,
                )
            )

            response.raise_for_status()

            envelope = (
                response.json()
            )

        raw = (
            envelope.get(
                "response"
            )
        )

        if not raw:
            raise ValueError(
                "Ollama returned an empty "
                "Grill-Me review."
            )

        parsed = (
            json.loads(
                raw
            )
        )

        context = (
            _safe_context(
                parsed,
                answers,
            )
        )

        logger.info(
            "Grill-Me semantic review complete: "
            "event=%s theme=%s reference=%s questions=%s",
            context.get(
                "event_type_interpretation"
            ),
            context.get(
                "theme_interpretation"
            ),
            context.get(
                "reference_name"
            ),
            len(
                context.get(
                    "clarification_questions",
                    [],
                )
            ),
        )

        return context

    except (
        httpx.HTTPError,
        ValueError,
        TypeError,
    ) as error:
        logger.warning(
            "Grill-Me semantic review unavailable: %s",
            error,
        )

        fallback = dict(
            DEFAULT_CONTEXT
        )

        fallback[
            "background_generation_summary"
        ] = (
            "Use the submitted event type, theme, colours "
            "and creative description as provided. "
            "Generate background artwork only."
        )

        return (
            _safe_context(
                fallback,
                answers,
            )
        )
