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


def _detailed_visual_prose(value: str, minimum_words: int = 16) -> bool:
    """Measure supplied detail without interpreting its subject or event type."""
    return len(re.findall(r"\b[\w'-]+\b", value)) >= minimum_words


def _grounded_visual_structure(context: dict | None) -> bool:
    if not context:
        return False
    semantics = context.get("visual_semantics") or {}
    provenance = context.get("semantic_provenance") or {}
    fields = ("motif_descriptors", "motif_structure_description",
              "composition_affinities", "material_descriptors", "texture_descriptors")
    populated = [semantics.get(field) for field in fields
                 if semantics.get(field) and provenance.get(field) != "SAFE_FALLBACK"]
    scene = (semantics.get("scene_family") not in (None, "", "general")
             and provenance.get("scene_family") != "SAFE_FALLBACK")
    return scene and bool(populated) or len(populated) >= 2


def _background_context_needs_clarification(answers: dict, context: dict | None = None) -> bool:
    """Ask plainly when the brief lacks a concrete background idea."""
    inspiration = str(answers.get("background_inspiration") or "").strip()
    if inspiration:
        words = re.findall(r"\b[\w'-]+\b", inspiration)
        if len(words) >= 3 and ("inspired" not in inspiration.casefold() or len(words) >= 5):
            return False
    if _grounded_visual_structure(context):
        return False
    description = " ".join(str(answers.get(field) or "") for field in
                           ("creative_description", "theme_reference_treatment"))
    return not _detailed_visual_prose(description)


_REFERENCE_SOURCE_FIELDS = ("theme", "creative_description", "background_inspiration")
_UNRESOLVED_REFERENCE_PATTERN = re.compile(
    r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)+[- ]inspired\b|"
    r"\b[A-Z][\w]*-[A-Z][\w]*-inspired\b|"
    r"\binspired by [A-Z][\w-]*(?:\s+[A-Z][\w-]*)+\b"
)


def _unresolved_reference_fields(answers: dict) -> list[str]:
    """Locate ambiguous source fields without guessing a reference identity."""
    return [field for field in _REFERENCE_SOURCE_FIELDS
            if _UNRESOLVED_REFERENCE_PATTERN.search(str(answers.get(field) or ""))]


def _unresolved_reference_like(answers: dict) -> bool:
    return bool(_unresolved_reference_fields(answers))


_TREATMENT_EXCLUSION = re.compile(
    r"\b(?:do not|don't|never|avoid|without|must not|should not|no)\b", re.I
)
_TRAIT_FILLER = {"a", "an", "and", "as", "background", "design", "for", "in", "into",
                 "motif", "motifs", "of", "or", "pattern", "patterns", "reference",
                 "should", "the", "to", "use", "visual", "with"}


def _treatment_parts(treatment: str, reference_name: str) -> tuple[str, list[str]]:
    """Separate selected visual traits from explicit bans without interpreting a theme."""
    approved, excluded = [], []
    reference_pattern = _reference_pattern(reference_name)
    for clause in re.split(r"[.;]+", treatment):
        clause = clause.strip()
        if not clause:
            continue
        match = _TREATMENT_EXCLUSION.search(clause)
        if match:
            selected = re.sub(r"\b(?:but|and)\s*$", "", clause[:match.start()].strip(), flags=re.I).strip()
            if selected:
                approved.append(selected)
            objects = re.sub(r"^(?:add|depict|draw|include|show|use|render)\s+", "",
                             clause[match.end():].strip(), flags=re.I)
            for object_name in re.split(r",|\s+or\s+", objects):
                object_name = re.sub(r"^(?:any|the|a|an)\s+", "", object_name.strip(), flags=re.I)
                if reference_pattern and reference_pattern.search(object_name):
                    object_name = "depicted named reference"
                if object_name:
                    excluded.append(object_name)
        else:
            approved.append(clause)
    return "; ".join(approved), list(dict.fromkeys(excluded))


def _trait_terms(value: str) -> set[str]:
    def singular(token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and not token.endswith(("ss", "us")) and len(token) > 3:
            return token[:-1]
        return token

    return {singular(token)
            for token in re.findall(r"[a-z]+", value.casefold())
            if token not in _TRAIT_FILLER and len(token) > 2}


def _reference_pattern(reference_name: str) -> re.Pattern | None:
    """Match a detected name across the same separator variants as CandidateSpec."""
    parts = [part for part in re.split(r"[\W_]+", reference_name) if part]
    return (re.compile(r"(?<!\w)" + r"[\W_]*".join(map(re.escape, parts)) + r"(?!\w)", re.I)
            if parts else None)


def _without_reference_identity(value: str, pattern: re.Pattern | None) -> str:
    if not pattern:
        return value
    text = re.sub(r"\b(?:most\s+especially|especially|inspired\s+by)\s+" + pattern.pattern,
                  "", value, flags=re.I)
    text = re.sub(pattern.pattern + r"[\W_]*inspired\b", "", text, flags=re.I)
    text = pattern.sub("", text)
    return re.sub(r"\s+", " ", re.sub(r"\s+([,.;])", r"\1", text)).strip(" ,.;-_")


def _reference_redaction_shell(value: str) -> bool:
    """Recognize reference removal that left no independent visual direction."""
    text = re.sub(r"\s+", " ", value.casefold()).strip(" ,.;-_")
    if re.search(r"\b(?:with|featuring|including|using)\b", text):
        return False
    if re.fullmatch(r"(?:(?:a|an|the)\s+)?(?:inspired\s+)?(?:theme|style|background|design)\s+for\s+.+", text):
        return True
    words = re.findall(r"[a-z]+", text)
    return bool(words) and set(words) <= {
        "a", "an", "the", "use", "apply", "inspired", "by", "theme", "style",
        "background", "design", "visual", "trait", "traits", "colour", "colours",
        "color", "colors", "and",
    }


def _selected_reference_trait(phrase: str, approved: str, excluded: list[str]) -> bool:
    terms = _trait_terms(phrase)
    if any(terms & _trait_terms(item) for item in excluded):
        return False
    return bool(terms) and terms <= _trait_terms(approved)


def _independently_grounded(phrase: str, key: str, answers: dict, review: dict,
                             pattern: re.Pattern | None, excluded: list[str]) -> bool:
    """Trust user wording or a review citation to non-reference user context."""
    sources = ("event_type", "age_group", "audience_context", "mood", "creative_description",
               "background_inspiration", "lighting_language", "material_language",
               "texture_language", "symbolic_motifs", "environmental_motifs")
    independent = " ".join(_without_reference_identity(str(answers.get(field) or ""), pattern)
                           for field in sources)
    if _selected_reference_trait(phrase, independent, excluded):
        return True
    grounding = review.get("semantic_grounding")
    evidence = grounding.get(key) if isinstance(grounding, dict) else None
    quote = evidence.get(phrase) if isinstance(evidence, dict) else None
    return bool(isinstance(quote, str) and len(quote.strip()) >= 4
                and _without_reference_identity(quote, pattern).casefold() == quote.casefold()
                and quote.casefold() in independent.casefold())


def renderable_brief(brief: dict) -> dict:
    """Keep only explicit safe visual intent when a reference stayed unresolved."""
    context = brief.get("grill_me_context") or {}
    if context.get("reference_detected") and brief.get("theme_reference_treatment"):
        name = str(context.get("reference_name") or "").strip()
        pattern = _reference_pattern(name)
        approved, _ = _treatment_parts(str(brief["theme_reference_treatment"]), name)
        approved = _without_reference_identity(approved, pattern)
        if _reference_redaction_shell(approved):
            approved = ""
        result = dict(brief)
        result["theme_reference_treatment"] = approved
        if pattern:
            for field in _REFERENCE_SOURCE_FIELDS:
                value = str(result.get(field) or "")
                cleaned = _without_reference_identity(value, pattern)
                result[field] = "" if _reference_redaction_shell(cleaned) else cleaned
        if not result.get("background_inspiration") and approved and not brief.get("background_inspiration"):
            result["background_inspiration"] = approved
        return result
    if context.get("reference_rendering_mode") != "safe_treatment_only":
        return brief

    fields = set(context.get("unresolved_reference_fields") or ())
    treatment = str(brief.get("theme_reference_treatment") or "").strip()
    result = dict(brief)
    for field in _REFERENCE_SOURCE_FIELDS:
        if field in fields:
            result[field] = ""
    if not result.get("background_inspiration"):
        result["background_inspiration"] = treatment

    def scrub(value):
        if isinstance(value, str):
            return "" if _UNRESOLVED_REFERENCE_PATTERN.search(value) else value
        if isinstance(value, list):
            cleaned = [scrub(item) for item in value]
            return [item for item in cleaned if item]
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    return scrub(result)


def model_facing_brief(brief: dict) -> dict:
    """Keep confirmed reference identity in metadata, outside the direction model input."""
    context = brief.get("grill_me_context") or {}
    name = str(context.get("reference_name") or "").strip() if context.get("reference_detected") else ""
    pattern = _reference_pattern(name)
    if not pattern:
        return dict(brief)

    def scrub(value):
        if isinstance(value, str):
            return _without_reference_identity(value, pattern)
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    result = scrub(brief)
    for field in ("background_inspiration", "theme_reference_treatment"):
        if _reference_redaction_shell(str(result.get(field) or "")):
            result[field] = ""
    if isinstance(result.get("visual_anchors"), list):
        result["visual_anchors"] = [item for item in result["visual_anchors"]
                                    if not isinstance(item, str) or not _reference_redaction_shell(item)]
    return result


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
    "reference_rendering_mode": "normal",
    "unresolved_reference_fields": [],
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
            (
                "For each semantic phrase grounded in event, audience or other user wording "
                "independently of a named reference, cite an exact excerpt from that user "
                "wording in semantic_grounding[field][phrase]. Omit unsupported citations."
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
            "semantic_grounding": {"field name": {"semantic phrase": "exact excerpt from non-reference user wording"}},
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


def _reference_colour_influence_confirmed(
    treatment: str,
    reference_name: str,
    colours: tuple[str, ...] = (),
) -> bool:
    """Confirm colour influence only from the approved positive treatment."""
    approved, _ = _treatment_parts(treatment, reference_name)
    pattern = _reference_pattern(reference_name)
    positive = _without_reference_identity(approved, pattern)
    colour_terms = _trait_terms(" ".join(colours))
    return bool(
        re.search(r"\b(?:colou?rs?|palette)\b", positive, re.I)
        or (_trait_terms(positive) & colour_terms)
    )


def _vague_reference_treatment(treatment: str, reference_name: str,
                               colours: tuple[str, ...] = ()) -> bool:
    approved, _ = _treatment_parts(treatment, reference_name)
    pattern = _reference_pattern(reference_name)
    positive = _without_reference_identity(approved, pattern)
    terms = _trait_terms(positive) - _trait_terms(" ".join(colours))
    concrete = terms - {
        "abstract", "colour", "color", "palette", "trait", "style", "vibe",
        "look", "feel", "theme", "inspired", "from", "must", "should",
        "influence", "only", "solely", "exclusively", "supplied", "reference",
        "other", "any", "few", "small", "amount", "like", "want", "see",
        "that", "consist", "based", "using", "some", "interesting", "generic",
        "superhero", "pattern", "shape", "form", "geometry", "line", "texture",
        "material", "atmosphere", "movement", "visual",
    }
    unspecified_traits = bool(
        re.search(r"\b(?:visual\s+)?traits?\b", positive, re.I)
    )
    explicit_colour_only = bool(
        re.search(r"\b(?:only|solely|exclusively)\b", positive, re.I)
        and _reference_colour_influence_confirmed(treatment, reference_name, colours)
        and not unspecified_traits
    )
    return not concrete and not explicit_colour_only

def _safe_inferred_reference_phrase(phrase: str, reference_name: str) -> bool:
    if _INFERRED_ASSET.search(phrase):
        return False
    pattern = _reference_pattern(reference_name)
    return not (pattern and pattern.search(phrase))


def _explicit_spatial_treatment(treatment: str) -> list[str]:
    """Normalize only viewpoint and movement actually stated by the user."""
    text = treatment.casefold()
    result = []
    if re.search(r"\bfirst[- ]person\b|\b(?:viewer|camera)\s+(?:is\s+)?(?:moving|travelling|traveling)", text):
        result.append("immersive first-person viewpoint")
    if re.search(r"\b(?:view from above|aerial view|overhead view)\b", text):
        result.append("aerial viewpoint")
    if re.search(r"\b(?:swing\w*|fly\w*|travel\w*|travers\w*|glid\w*|rush\w*|"
                 r"mov\w*\s+(?:through|across|past|between|over))\b", text):
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
    *,
    semantic_review_available: bool = True,
) -> dict:
    if not isinstance(value, dict):
        value = {}

    context = dict(DEFAULT_CONTEXT)
    context.update({key: item for key, item in value.items() if key in context})
    # The rendering state is established locally from this review outcome.
    context["reference_rendering_mode"] = "normal"
    context["unresolved_reference_fields"] = []
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
    if event and context["event_type_status"] in {"clear", "known"}:
        context["event_type_status"] = "known"
        context["event_type_interpretation"] = context["event_type_interpretation"] or event
    elif event and context["event_type_status"] == "unknown":
        context.update(event_type_status="provided", event_type_interpretation=event)
    if theme and context.get("theme_status") != "known":
        context.update(theme_status="known", theme_interpretation=theme)
    exclusions = []
    if context.get("reference_detected"):
        reference_name = str(context.get("reference_name") or "").strip()
        treatment = _theme_reference_treatment(answers)
        vague = _vague_reference_treatment(treatment, reference_name,
                                           tuple(str(answers.get(key) or "") for key in
                                                 ("primary_colour", "secondary_colour")))
        explicit_spatial = _explicit_spatial_treatment(treatment) if not vague else []
        approved, exclusions = _treatment_parts(treatment, reference_name)
        pattern = _reference_pattern(reference_name)
        semantics = context["visual_semantics"]
        for key, content in semantics.items():
            if key == "scene_family":
                if any(_trait_terms(item) & _trait_terms("architecture buildings interior")
                       for item in exclusions):
                    semantics[key] = "general"
                continue
            explicit = isinstance(supplied_semantics, dict) and key in supplied_semantics
            if isinstance(content, list):
                semantics[key] = [phrase for phrase in content
                                  if _safe_inferred_reference_phrase(phrase, reference_name)
                                  and (key == "incompatible_treatments" or
                                       (not any(_trait_terms(phrase) & _trait_terms(item)
                                                for item in exclusions) and
                                        (explicit or _independently_grounded(
                                            phrase, key, answers, value, pattern, exclusions) or
                                         _selected_reference_trait(phrase, approved, exclusions))))]
            elif content and (not _safe_inferred_reference_phrase(content, reference_name)
                              or any(_trait_terms(content) & _trait_terms(item) for item in exclusions)
                              or (not explicit and not _independently_grounded(
                                  content, key, answers, value, pattern, exclusions)
                                  and not _selected_reference_trait(content, approved, exclusions))):
                semantics[key] = ""
        semantics["incompatible_treatments"] = list(dict.fromkeys(
            [*semantics["incompatible_treatments"], *exclusions]))[:8]
        context["visual_semantics"] = semantics
        if explicit_spatial:
            semantics["composition_affinities"] = list(dict.fromkeys(
                [*semantics["composition_affinities"], *explicit_spatial]))[:8]
    context["semantic_provenance"] = {
        key: ("SAFE_FALLBACK" if not content or (key == "scene_family" and content == "general"
                                    and not (isinstance(supplied_semantics, dict) and key in supplied_semantics)) else
              "GRILL_ME_CLARIFIED" if key == "incompatible_treatments" and exclusions else
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
        _background_context_needs_clarification(answers, context)
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
        and (not treatment or _vague_reference_treatment(
            treatment, str(context.get("reference_name") or ""),
            tuple(str(answers.get(key) or "") for key in ("primary_colour", "secondary_colour"))))
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
        colours = tuple(
            str(answers.get(key) or "")
            for key in ("primary_colour", "secondary_colour")
        )
        colour_influence_confirmed = _reference_colour_influence_confirmed(
            treatment,
            str(reference_name),
            colours,
        )

        questions.insert(
            0,
            {
                "field": (
                    "theme_reference_treatment"
                ),
                "question": (
                    "You've confirmed the colour influence. Which other visual traits "
                    "should shape the background, such as patterns, shapes, textures, "
                    "atmosphere or movement? If you want colour-only influence, say so explicitly."
                    if colour_influence_confirmed else
                    f"Which concrete visual traits from {reference_name} should shape the background, "
                    "such as patterns, shapes, textures, atmosphere or movement? "
                    "If you want the reference to influence only the colours, say that explicitly."
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

    unresolved_fields = (_unresolved_reference_fields(answers)
                         if not semantic_review_available and not reference_detected else [])
    treatment_sufficient = _detailed_visual_prose(treatment, minimum_words=10)
    unresolved_reference_question = bool(unresolved_fields and not treatment_sufficient)
    if unresolved_fields and treatment_sufficient:
        context["reference_rendering_mode"] = "safe_treatment_only"
        context["unresolved_reference_fields"] = unresolved_fields
    if unresolved_reference_question:
        context["reference_needs_clarification"] = True
        questions = [item for item in questions
                     if item["field"] != "theme_reference_treatment"]
        questions.insert(0, {
            "field": "theme_reference_treatment",
            "question": ("Your theme may refer to a particular visual reference. "
                         "Which colours, patterns, shapes, atmosphere or other visual traits "
                         "should influence the background?"),
            "reason": "The visual reference could not be resolved during semantic review.",
        })

    background_fields = {"background_inspiration"}
    has_background_question = any(
        item["field"] in background_fields
        for item in questions
    )

    if (
        _background_context_needs_clarification(answers, context)
        and not unresolved_reference_question
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
                semantic_review_available=False,
            )
        )
