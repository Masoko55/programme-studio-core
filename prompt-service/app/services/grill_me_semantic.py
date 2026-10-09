import json
import logging

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
                "Examples of established event categories include "
                "birthday party, kids party, bridal shower, baby shower, "
                "wedding, graduation, black-tie event, awards ceremony, "
                "corporate function and similar event types."
            ),
            (
                "Determine whether the theme references a recognizable "
                "fictional character, franchise, celebrity, brand, era, "
                "cultural property, sports property or visual style."
            ),
            (
                "Examples include Spider-Man, Batman, Barbie, Disney, "
                "Marvel, Harry Potter, Formula 1, Great Gatsby, "
                "Roaring Twenties and similar references."
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
                "as 'fun superhero birthday celebration' is sufficient "
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


def _safe_context(
    value,
    answers: dict,
) -> dict:
    if not isinstance(
        value,
        dict,
    ):
        value = {}

    context = dict(
        DEFAULT_CONTEXT
    )

    context.update(
        {
            key: item
            for key, item
            in value.items()
            if key in context
        }
    )

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

    required_constraints = [
        "background only",
        "no people",
        "no characters",
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
        json.JSONDecodeError,
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