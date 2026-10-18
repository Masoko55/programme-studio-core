import json
import logging
import re

from app.config.settings import settings

from app.schemas.creative_direction import (
    CreativeDirectionOutput,
    LayoutGuidance,
    LayoutZone,
)

from app.services.ollama import (
    generate_text,
)
from app.services.grill_me_semantic import grounded_scene_family
from app.services.direction_diversity import (
    locked_structural_fields,
    materialize_locked_structural_fields,
)


SKY_BLUE = "sky blue"


logger = logging.getLogger(
    "uvicorn.error"
)


POSITIVE_PROMPT_FORBIDDEN_TERMS = (
    "typography",
    "lettering",
    "watermark",
    "readable text",
    "written text",
    "text",
    "words",
    "writing",
    "logo",
    "headshot",
    "portrait",
    "person",
    "people",
    "human",
    "face",
    "figure",
    "title",
    "programme",
    "program",
    "agenda",
    "schedule",
    "display",
    "character",
    "characters",
    "hero",
    "heroes",
    "superhero",
    "superheroes",
    "masked person",
    "masked character",
    "costume",
    "costumed",
)


COLOUR_ALIASES = {
    "gray": "grey",
    "golden": "gold",
    "navy blue": "navy",
    "royal": "royal blue",
    "sky": SKY_BLUE,
    "baby blue": SKY_BLUE,
    "light blue": SKY_BLUE,
    "dark blue": "navy",
    "hotpink": "hot pink",
    "rose pink": "rose",
    "rose-pink": "rose",
    "fuchsia": "magenta",
    "lilac": "lavender",
    "wine": "burgundy",
    "wine red": "burgundy",
    "rose-gold": "rose gold",
}


def resolve_palette_mode(brief: dict) -> str:
    """Use the same dominant/strict distinction as image validation."""
    requested = str(brief.get("palette_mode") or "").lower()
    if requested in {"strict", "monochrome", "achromatic"}:
        return "strict" if requested == "strict" else "achromatic"
    description = " ".join(str(brief.get(key) or "") for key in (
        "creative_description", "palette_relationship", "colour_palette",
    )).lower()
    if re.search(r"\b(?:only (?:these |the )?colou?rs|use only\b|no third colou?r|strictly monochrome|exclusive palette)\b", description):
        return "strict"
    return "dominant"


def direction_structural_summary(direction: CreativeDirectionOutput, brief: dict) -> dict:
    """Carry a previous direction's spatial plan into later direction prompts."""
    return {
        "direction_id": direction.direction_id,
        "composition_strategy": direction.composition_strategy,
        "focal_strategy": direction.focal_strategy,
        "depth_strategy": direction.depth_strategy,
        "border_or_perimeter_strategy": direction.border_or_perimeter_strategy,
        "title_safe_region_strategy": direction.title_safe_region_strategy,
        "programme_safe_region_strategy": direction.programme_safe_region_strategy,
        "background_inspiration": brief.get("background_inspiration"),
    }



def normalize_colour(
    value: str | None,
) -> str | None:
    if not value:
        return None

    normalized = (
        value
        .strip()
        .lower()
        .replace(
            "_",
            " ",
        )
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return (
        COLOUR_ALIASES.get(
            normalized,
            normalized,
        )
    )


def requested_palette(
    brief: dict,
) -> list[str]:
    colours = []

    for item in brief.get("palette") or []:
        value = item.get("colour") if isinstance(item, dict) else item
        colour = normalize_colour(str(value)) if value else None
        if colour and colour not in colours:
            colours.append(colour)

    for key in (
        "primary_colour",
        "secondary_colour",
    ):
        colour = (
            normalize_colour(
                brief.get(
                    key
                )
            )
        )

        if (
            colour
            and colour
            not in colours
        ):
            colours.append(
                colour
            )

    return colours


def positive_palette_description(
    brief: dict,
) -> str:
    colours = (
        requested_palette(
            brief
        )
    )

    if not colours:
        return (
            "A restrained cohesive colour palette "
            "with balanced tonal variation."
        )

    primary = (
        colours[
            0
        ]
    )

    secondary = (
        colours[
            1
        ]
        if len(
            colours
        ) > 1
        else None
    )

    mode = resolve_palette_mode(
        brief
    )

    if (
        set(
            colours
        )
        == {
            "black",
            "white",
        }
    ):
        return (
            "A strictly achromatic palette of black and white, "
            "with tonal variation inside those requested neutral colours."
        )

    if secondary:
        relationship = str(
            brief.get(
                "palette_relationship"
            )
            or (
                "the requested colour roles remain "
                "visually clear"
            )
        )

        if mode == "strict":
            return (
                f"A strict {', '.join(colours)} palette; "
                f"{relationship}. Only the requested colour families "
                "form the visible palette, apart from microscopic "
                "anti-aliasing tolerance."
            )

        return (
            f"A {', '.join(colours)} palette; "
            f"{relationship}. Requested colours control the "
            "main surfaces and visual mass. Small bounded achromatic shadows, "
            "highlights and fine structural linework may support the design, "
            "but must not become a background colour field or displace the "
            "requested colours. Third chromatic design colour presence "
            "remains negligible."
        )

    if mode == "strict":
        return (
            f"A strict {primary} palette. Only the requested colour families "
            f"are represented, with {primary} remaining visually exclusive "
            "apart from microscopic anti-aliasing tolerance."
        )

    return (
        f"A controlled {primary} palette with {primary} clearly "
        "dominant, supported by bounded achromatic linework, "
        "shadows and highlights. third chromatic design colour "
        "presence remains negligible."
    )

def palette_negative(
    brief: dict,
) -> str:
    colours = (
        requested_palette(
            brief
        )
    )

    if not colours:
        return (
            "off-palette colours, colour drift, "
            "unrequested accent colours, multicolour palette, "
            "unrelated hues"
        )

    if resolve_palette_mode(brief) != "strict":
        return ("unrequested chromatic design colours, colour drift, unrelated hues, "
                "large neutral areas that displace the requested colours")

    return (
        "off-palette colours, colour drift, "
        "unrequested accent colours, multicolour palette, "
        "unrelated hues, any neutral or chromatic colour family "
        "outside the requested palette. "
        "The only requested colour families are: "
        + ", ".join(
            colours
        )
    )


def _asset_zone(
    placement: str,
) -> LayoutZone:
    normalized = (
        str(
            placement
            or "center"
        )
        .strip()
        .lower()
    )

    aliases = {
        "centre": "center",
        "middle": "center",
        "top-left": "left",
        "top left": "left",
        "top-center": "center",
        "top centre": "center",
        "top center": "center",
        "top-centre": "center",
        "top-right": "right",
        "top right": "right",
    }

    normalized = (
        aliases.get(
            normalized,
            normalized,
        )
    )

    positions = {
        "left": 0.08,
        "center": 0.39,
        "right": 0.70,
    }

    x = (
        positions.get(
            normalized,
            0.39,
        )
    )

    return (
        LayoutZone(
            x=x,
            y=0.045,
            width=0.22,
            height=0.16,
        )
    )


def asset_layout_description(
    brief: dict,
) -> str:
    """
    This description is for the LLM's layout reasoning only.

    IMPORTANT:
    It must never be appended directly to positive_prompt because
    it contains layout terminology such as title/programme/text,
    which the image-prompt validator intentionally forbids.
    """

    asset_type = (
        str(
            brief.get(
                "asset_type"
            )
            or "none"
        )
        .strip()
        .lower()
    )

    if (
        asset_type
        not in {
            "headshot",
            "logo",
        }
    ):
        return (
            "No uploaded asset is enabled. "
            "Use the standard layout zones."
        )

    placement = (
        str(
            brief.get(
                "asset_placement"
            )
            or "center"
        )
        .strip()
        .lower()
    )

    return (
        f"The {asset_type} occupies a reserved top-{placement} "
        "layout zone. All other content zones begin underneath "
        "the reserved asset area."
    )


def safe_layout(
    brief: dict,
) -> LayoutGuidance:
    """
    Layout coordinates are deterministic.

    We do not trust an LLM to calculate asset placement coordinates.
    """

    asset_type = (
        str(
            brief.get(
                "asset_type"
            )
            or "none"
        )
        .strip()
        .lower()
    )

    asset_placement = (
        str(
            brief.get(
                "asset_placement"
            )
            or "center"
        )
        .strip()
        .lower()
    )

    if (
        asset_type
        in {
            "headshot",
            "logo",
        }
    ):
        asset_zone = (
            _asset_zone(
                asset_placement
            )
        )

        #
        # Asset:
        # y = 0.045 -> 0.205
        #
        # Content begins below it.
        #
        title_zone = (
            LayoutZone(
                x=0.10,
                y=0.245,
                width=0.80,
                height=0.12,
            )
        )

        programme_zone = (
            LayoutZone(
                x=0.10,
                y=0.43,
                width=0.80,
                height=0.47,
            )
        )

        if (
            asset_type
            == "headshot"
        ):
            return (
                LayoutGuidance(
                    title_zone=(
                        title_zone
                    ),
                    programme_zone=(
                        programme_zone
                    ),
                    headshot_zone=(
                        asset_zone
                    ),
                    logo_zone=None,
                )
            )

        return (
            LayoutGuidance(
                title_zone=(
                    title_zone
                ),
                programme_zone=(
                    programme_zone
                ),
                headshot_zone=None,
                logo_zone=(
                    asset_zone
                ),
            )
        )

    return (
        LayoutGuidance(
            title_zone=(
                LayoutZone(
                    x=0.10,
                    y=0.08,
                    width=0.80,
                    height=0.12,
                )
            ),
            programme_zone=(
                LayoutZone(
                    x=0.10,
                    y=0.47,
                    width=0.80,
                    height=0.40,
                )
            ),
            headshot_zone=None,
            logo_zone=None,
        )
    )


def direction_configuration(
    direction_id: str,
) -> tuple[
    str,
    str,
]:
    normalized = (
        direction_id
        .lower()
    )

    if (
        normalized
        not in {
            "a",
            "b",
            "c",
        }
    ):
        raise ValueError(
            "Unsupported creative direction: "
            f"{direction_id}"
        )

    model = getattr(
        settings,
        f"direction_{normalized}_model",
    )

    role = getattr(
        settings,
        f"direction_{normalized}_role",
    )

    return (
        model,
        role,
    )


def composition_variant(
    direction_id: str,
) -> str:
    return (
        "Choose a composition, hierarchy, focal treatment, depth and safe-space "
        "strategy that follows this brief. Make it materially distinct from "
        "the other directions without assigning a fixed style to this slot."
    )


def _background_inspiration(brief: dict) -> str:
    """Return the user's plain-language visual idea for direction prompts."""
    return str(
        brief.get("background_inspiration")
        or brief.get("grill_me_context", {}).get("background_generation_summary")
        or "an elegant abstract event background"
    ).strip()


def _list_values(brief: dict, key: str) -> list[str]:
    value = brief.get(key) or []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def _is_negative_instruction_clause(
    value: str,
) -> bool:
    """Return True when prose is an exclusion/instruction rather than visual description."""
    return bool(
        re.search(
            r"\b(?:do not|don't|never|avoid|without|must not|should not|no)\b",
            str(value),
            re.IGNORECASE,
        )
    )


def _reference_name(
    brief: dict,
) -> str:
    context = (
        brief.get(
            "grill_me_context"
        )
        or {}
    )

    if not context.get(
        "reference_detected"
    ):
        return ""

    return str(
        context.get(
            "reference_name"
        )
        or ""
    ).strip()


def _contains_reference_name(
    value: str,
    brief: dict,
) -> bool:
    name = _reference_name(
        brief
    )

    if not name:
        return False

    return bool(
        re.search(
            rf"\b{re.escape(name)}\b",
            str(value),
            re.IGNORECASE,
        )
    )


def _semantic_values(
    brief: dict,
    key: str,
) -> list[str]:
    semantics = (
        brief.get(
            "visual_semantics"
        )
        or (
            brief.get(
                "grill_me_context"
            )
            or {}
        ).get(
            "visual_semantics"
        )
        or {}
    )

    value = semantics.get(
        key
    )

    if isinstance(
        value,
        list,
    ):
        return [
            str(item).strip()
            for item
            in value
            if str(item).strip()
        ]

    if value:
        return [
            str(value).strip()
        ]

    return []


def _renderable_subject_description(
    brief: dict,
) -> str:
    """Describe the renderable subject without leaking a named reference."""
    if _reference_name(
        brief
    ):
        parts = []

        for key in (
            "motif_structure_description",
            "motif_descriptors",
            "composition_affinities",
            "atmosphere_descriptors",
        ):
            for value in _semantic_values(
                brief,
                key,
            ):
                phrase = _safe_visual_phrase(
                    value
                )

                if (
                    phrase
                    and not _contains_reference_name(
                        phrase,
                        brief,
                    )
                ):
                    parts.append(
                        phrase
                    )

        if parts:
            return ", ".join(
                dict.fromkeys(
                    parts
                )
            )

    inspiration = (
        _background_inspiration(
            brief
        )
    )

    if not _contains_reference_name(
        inspiration,
        brief,
    ):
        return inspiration

    return (
        "abstract brief-specific visual geometry, "
        "materials, atmosphere and spatial structure"
    )


def _safe_visual_phrase(value: str | None) -> str:
    """Keep generic visual-language prose while excluding non-renderable instructions."""
    if not value:
        return ""

    clauses = []
    for clause in re.split(r"[,;.!?]+", str(value)):
        clause = " ".join(clause.split())
        if not clause:
            continue
        if any(
            re.search(rf"\b{re.escape(term)}\b", clause, re.IGNORECASE)
            for term in POSITIVE_PROMPT_FORBIDDEN_TERMS
        ):
            continue
        if _is_negative_instruction_clause(
            clause
        ):
            continue
        clauses.append(clause)

    return ", ".join(clauses)


def reference_visual_language(
    brief: dict,
) -> str:
    """Build descriptive visual language without named-reference or exclusion leakage."""
    parts = []

    semantics = (
        brief.get(
            "visual_semantics"
        )
        or (
            brief.get(
                "grill_me_context"
            )
            or {}
        ).get(
            "visual_semantics"
        )
        or {}
    )

    semantic_fields = (
        (
            "motif_descriptors",
            "motifs",
        ),
        (
            "motif_structure_description",
            "motif structure",
        ),
        (
            "atmosphere_descriptors",
            "atmosphere",
        ),
        (
            "lighting_descriptors",
            "lighting",
        ),
        (
            "material_descriptors",
            "materials",
        ),
        (
            "texture_descriptors",
            "texture",
        ),
        (
            "composition_affinities",
            "composition",
        ),
    )

    provenance = ((brief.get("semantic_contract") or {}).get("semantic_provenance") or
                  (brief.get("grill_me_context") or {}).get("semantic_provenance") or {})

    for key, label in semantic_fields:
        if key in {"lighting_descriptors", "material_descriptors", "texture_descriptors"} and (
            provenance.get(key) not in {"USER_EXPLICIT", "GRILL_ME_CLARIFIED"}
        ):
            continue
        value = semantics.get(
            key
        )

        values = (
            value
            if isinstance(
                value,
                list,
            )
            else [
                value
            ]
        )

        phrases = []

        for item in values:
            phrase = _safe_visual_phrase(
                item
            )

            if (
                phrase
                and not _contains_reference_name(
                    phrase,
                    brief,
                )
            ):
                phrases.append(
                    phrase
                )

        if phrases:
            parts.append(
                f"{label}: "
                + ", ".join(
                    phrases
                )
            )

    #
    # For briefs without a detected named reference, preserve the
    # user's generic clarified visual fields as before.
    #
    if not _reference_name(
        brief
    ):
        for key, label in (
            (
                "theme_reference_treatment",
                "reference treatment",
            ),
            (
                "visual_style",
                "visual style",
            ),
            (
                "material_language",
                "materials",
            ),
            (
                "texture_language",
                "texture",
            ),
            (
                "lighting_language",
                "lighting",
            ),
            (
                "mood",
                "mood",
            ),
        ):
            phrase = _safe_visual_phrase(
                brief.get(
                    key
                )
            )

            if phrase:
                parts.append(
                    f"{label}: {phrase}"
                )

        for key, label in (
            (
                "visual_anchors",
                "visual anchors",
            ),
            (
                "symbolic_motifs",
                "symbolic motifs",
            ),
            (
                "environmental_motifs",
                "environmental motifs",
            ),
        ):
            phrases = [
                phrase
                for phrase
                in (
                    _safe_visual_phrase(
                        value
                    )
                    for value
                    in _list_values(
                        brief,
                        key,
                    )
                )
                if phrase
            ]

            if phrases:
                parts.append(
                    f"{label}: "
                    + ", ".join(
                        phrases
                    )
                )

    return (
        "Clarified visual language: "
        + "; ".join(
            dict.fromkeys(
                parts
            )
        )
        + ". "
        if parts
        else ""
    )

def background_design_summary(
    brief: dict,
) -> str:
    """Describe the requested artwork without imposing a stock layout."""
    return (
        "User background inspiration: "
        + _background_inspiration(brief)
        + ". "
        + reference_visual_language(brief)
        + "Create original background artwork that follows the requested subject, "
        + "aesthetic, palette and event context. Preserve usable negative space "
        + "for later typography, but choose its shape and placement from the "
        + "brief and the creative direction rather than a boilerplate frame. "
        + "Layout contract: "
        + asset_layout_description(brief)
        + "."
    )


def background_positive_requirements(
    brief: dict,
) -> str:
    """Return descriptive visual requirements only."""
    subject = (
        _renderable_subject_description(
            brief
        )
    )

    return (
        "Visual subject: "
        + subject
        + ". "
        + reference_visual_language(
            brief
        )
        + "Original brief-specific composition, coherent visual "
        + "hierarchy, intentional calm negative space, balanced "
        + "depth and gentle tonal modelling."
    )

def background_negative_constraints() -> str:
    return "unrelated generic imagery, crowded centre, readable text, people, characters"


def fallback_prompt(
    brief: dict,
    role: str,
) -> str:
    palette = (
        positive_palette_description(
            brief
        )
    )

    subject = (
        _renderable_subject_description(
            brief
        )
    )

    return (
        f"{role} A4 portrait event-background artwork, "
        f"{palette} "
        "Brief-specific composition, coherent visual hierarchy, "
        "intentional calm negative space and subject-appropriate depth. "
        + subject
        + "."
    )

_MATERIAL_WORDS = re.compile(
    r"\b(?:concrete|parchment|paper|metal(?:lic)?|wood(?:en)?|velvet|"
    r"marble|plastic|glass|fabric|stone|leather|silk|clay|ceramic)\b", re.I,
)
_TEXTURE_WORDS = re.compile(
    r"\b(?:texture[ds]?|glossy|matte|etched|grainy|embossed)\b", re.I,
)
_LIGHTING_WORDS = re.compile(
    r"\b(?:rim lighting|spotlight|backlit|neon lighting|studio lighting)\b", re.I,
)
_APPEARANCE_WORDS = re.compile(
    r"\b(?:atmospheric perspective|watercolou?r|photorealistic|grunge|"
    r"dark (?:atmospheric )?(?:background|field|wash)|"
    r"light (?:background|field|wash)|(?:muted|desaturated|monochrome) (?:background|palette|field))\b", re.I,
)
_NAMED_COLOURS = re.compile(
    r"\b(?:black|white|grey|gray|charcoal|brown|green|forest green|orange|"
    r"purple|violet|yellow|pink|magenta|cyan|teal|turquoise|beige|cream|"
    r"navy|burgundy|maroon|blue|red|gold|silver)\b", re.I,
)


def _unsupported_treatment(clause: str, brief: dict) -> bool:
    """Keep concrete treatments only when the user's brief supports them."""
    semantics = brief.get("visual_semantics") or {}
    explicit_text = " ".join(str(brief.get(key) or "") for key in (
        "theme", "background_inspiration", "creative_description", "theme_reference_treatment",
        "material_language", "texture_language", "lighting_language", "visual_style",
    )).casefold()
    provenance = ((brief.get("semantic_contract") or {}).get("semantic_provenance") or
                  (brief.get("grill_me_context") or {}).get("semantic_provenance") or {})
    for key in ("material_descriptors", "texture_descriptors", "lighting_descriptors"):
        if provenance.get(key) in {"USER_EXPLICIT", "GRILL_ME_CLARIFIED"}:
            explicit_text += " " + " ".join(str(value) for value in (semantics.get(key) or [])).casefold()
    for pattern in (_MATERIAL_WORDS, _TEXTURE_WORDS, _LIGHTING_WORDS, _APPEARANCE_WORDS):
        if any(match.group().casefold() not in explicit_text and not (
            match.group().casefold() == "dark atmospheric background" and "dark background" in explicit_text
        ) for match in pattern.finditer(clause)):
            return True
    requested = {normalize_colour(colour) for colour in requested_palette(brief)}
    for match in _NAMED_COLOURS.finditer(clause):
        colour = normalize_colour(match.group())
        if colour not in requested and match.group().casefold() not in explicit_text:
            return True
    return False


def sanitize_positive_prompt(
    value: str,
    brief: dict,
    role: str,
) -> str:
    clauses = re.split(
        r"[,;.!?]+",
        value,
    )

    clean = []

    for clause in clauses:
        clause = (
            clause.strip()
        )

        if not clause:
            continue

        blocked = any(
            re.search(
                rf"\b{re.escape(term)}\b",
                clause,
                re.IGNORECASE,
            )
            for term
            in POSITIVE_PROMPT_FORBIDDEN_TERMS
        )

        negative_instruction = bool(
            re.search(
                r"\b(?:do not|don't|no|without|avoid)\b",
                clause,
                re.IGNORECASE,
            )
        )

        named_reference = (
            _contains_reference_name(
                clause,
                brief,
            )
        )

        unsupported_treatment = _unsupported_treatment(clause, brief)

        if (
            not blocked
            and not negative_instruction
            and not named_reference
            and not unsupported_treatment
        ):
            clean.append(
                clause
            )

    result = (
        ", ".join(
            clean
        )
    )

    if (
        len(
            result
        )
        < 40
    ):
        result = (
            fallback_prompt(
                brief,
                role,
            )
        )

    palette = (
        positive_palette_description(
            brief
        )
    )

    result = (
        result.rstrip(
            ". "
        )
        + ". "
        + palette
        + " "
        + background_positive_requirements(
            brief
        )
        + " Theme references are represented through brief-grounded abstract "
        + "geometry, patterns, spatial structure, atmosphere "
        + "and colour relationships. Architecture, environment, materials, "
        + "textures and lighting are used only when supported by the brief."
    )

    #
    # Final defensive sanitisation.
    #
    # Because deterministic content is appended above, run another
    # pass to make sure forbidden rendering concepts were not
    # accidentally reintroduced.
    #
    clauses = re.split(
        r"[,;.!?]+",
        result,
    )

    clean_final = []
    seen_clauses = set()

    for clause in clauses:
        clause = (
            clause.strip()
        )

        if not clause:
            continue

        blocked = any(
            re.search(
                rf"\b{re.escape(term)}\b",
                clause,
                re.IGNORECASE,
            )
            for term
            in POSITIVE_PROMPT_FORBIDDEN_TERMS
        )

        negative_instruction = (
            _is_negative_instruction_clause(
                clause
            )
        )

        named_reference = (
            _contains_reference_name(
                clause,
                brief,
            )
        )

        key = re.sub(
            r"\s+",
            " ",
            clause,
        ).casefold()

        if (
            not blocked
            and not negative_instruction
            and not named_reference
            and key not in seen_clauses
        ):
            clean_final.append(
                clause
            )
            seen_clauses.add(
                key
            )

    final_prompt = (
        ", ".join(
            clean_final
        )
        .strip()
    )

    if (
        len(
            final_prompt
        )
        < 40
    ):
        raise ValueError(
            "Sanitised positive prompt became too short."
        )

    return (
        final_prompt
    )


def deduplicate_negative_prompt(
    value: str,
) -> str:
    """Keep model exclusions compact without dropping any unique restriction."""
    clauses = []
    seen = set()

    for clause in str(value or "").split(","):
        cleaned = " ".join(clause.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            clauses.append(cleaned)
            seen.add(key)

    return ", ".join(clauses)


def build_creative_direction_prompt(
    brief: dict,
    direction_id: str,
    role: str,
    correction_error: str | None = None,
    previous_directions: list[str] | None = None,
    diversity_constraints: dict | None = None,
) -> str:
    asset_type = (
        str(
            brief.get(
                "asset_type"
            )
            or "none"
        )
        .lower()
    )

    asset_enabled = (
        asset_type
        in {
            "headshot",
            "logo",
        }
    )

    requirements = [
        (
            "Use the supplied event brief as the source of truth."
        ),
        (
            "Create visual background artwork rather than a finished poster."
        ),
        (
            "Keep positive_prompt purely visual and descriptive."
        ),
        (
            "Never mention title, programme, text, logo, headshot, "
            "schedule, agenda or typography inside positive_prompt."
        ),
        (
            "Put exclusions and unwanted rendering concepts only "
            "inside negative_prompt."
        ),
        (
            "If the event brief uses a recognizable person, fictional "
            "character, superhero, mascot, celebrity or franchise, "
            "translate the reference only into brief-grounded abstract geometry, "
            "patterns, spatial structure, atmosphere and colour relationships. "
            "Use architecture, environment, materials, textures or lighting "
            "only when supported by the grounded semantic contract."
        ),
        (
            "Never place the referenced person or character itself "
            "inside positive_prompt."
        ),
        ("Respect the palette relationship specified by the user."),
        (positive_palette_description(brief)),
        (
            "Use strong tonal contrast without inventing an "
            "unrequested colour family."
        ),
        (
            "Follow background subject, style, motifs, exclusions "
            "and composition exactly."
        ),
        (
            "Apply this composition variation: "
            + composition_variant(
                direction_id
            )
        ),
        (
            "Do not use a stock programme template, universal border, fixed central "
            "panel, repeated side bars or a boilerplate composition. Invent the "
            "composition from this brief."
        ),
        (
            "The new direction must differ structurally from prior directions in at "
            "least two unlocked spatial decisions: composition family, viewpoint, "
            "focal band, mass distribution, major flow, depth, perimeter, motif placement "
            "or quiet-region geometry. Preserve every user-locked spatial requirement. "
            "Rewording the same arrangement does not count."
        ),
        ("Name the viewpoint and distribution of major visual masses in the strategy fields; "
         "use the user's requested subject in every direction; vary only unlocked camera and layout choices."),
        ("Vary only unlocked viewpoint, mass placement, depth and spatial flow. "
         "Do not invent material, texture or specific lighting treatments when the semantic contract leaves them empty. "
         "Do not introduce unsupported scene-defining objects. Keep each supplied motif at its stated "
         "importance: a subtle accent must not expand to cover the scene. Prefer a few coherent large "
         "forms over uniformly distributed micro-detail unless the brief explicitly requests density."),
        ("Describe the chosen composition, focal point, depth, perimeter treatment and separate calm title/programme areas in their strategy fields."),
        ("Use the prior direction concepts to choose a materially different arrangement, not just a new palette."),
        (
            "Preserve calm visual regions through the composition."
        ),
        (
            "Layout coordinates must follow the supplied layout contract."
        ),
        (
            "Return JSON matching the supplied output schema."
        ),
    ]

    if asset_enabled:
        requirements.extend(
            [
                (
                    "A reserved top layout zone is required for "
                    f"the selected {asset_type}."
                ),
                (
                    "All non-asset layout zones must begin below "
                    "the reserved top asset zone."
                ),
                (
                    "Do not describe the reserved asset inside "
                    "positive_prompt; it belongs only in layout_guidance."
                ),
            ]
        )

    event_brief = dict(brief)
    grounded_family = grounded_scene_family(brief)
    if isinstance(event_brief.get("visual_semantics"), dict):
        event_brief["visual_semantics"] = {**event_brief["visual_semantics"],
                                          "scene_family": grounded_family}
    if isinstance(event_brief.get("grill_me_context"), dict):
        event_brief["grill_me_context"] = dict(event_brief["grill_me_context"])
        context_semantics = event_brief["grill_me_context"].get("visual_semantics")
        if isinstance(context_semantics, dict):
            event_brief["grill_me_context"]["visual_semantics"] = {
                **context_semantics, "scene_family": grounded_family}
    if isinstance(event_brief.get("semantic_contract"), dict):
        event_brief["semantic_contract"] = {**event_brief["semantic_contract"],
                                            "scene_family": grounded_family}

    payload = {
        "task": (
            "Create one creative direction for an "
            "A4 portrait event background."
        ),
        "direction_id": (
            direction_id.upper()
        ),
        "composition_variant": (
            composition_variant(
                direction_id
            )
        ),
        "background_design_summary": (
            background_design_summary(
                brief
            )
        ),
        "background_positive_requirements": (
            background_positive_requirements(
                brief
            )
        ),
        "background_negative_constraints": (
            background_negative_constraints()
        ),
        "asset_layout_contract": {
            "enabled": (
                asset_enabled
            ),
            "asset_type": (
                asset_type
            ),
            "horizontal_placement": (
                brief.get(
                    "asset_placement"
                )
            ),
            "vertical_placement": (
                brief.get(
                    "asset_vertical_position"
                )
            ),
            "flow": (
                brief.get(
                    "text_flow"
                )
            ),
            "description": (
                asset_layout_description(
                    brief
                )
            ),
        },
        "creative_role": "Choose a brief-specific visual role and composition; no slot has a preset style.",
        "previous_direction_concepts": previous_directions or [],
        "event_brief": event_brief,
        "positive_palette_description": (
            positive_palette_description(
                brief
            )
        ),
        "negative_palette_constraints": (
            palette_negative(
                brief
            )
        ),
        "automatic_rendering_policy": {
            "high_contrast": True,
            "readable_overlay_regions": True,
            "background_only": True,
            "characters_allowed": False,
            "people_allowed": False,
            "asset_enabled": (
                asset_enabled
            ),
            "asset_position_is_top": (
                asset_enabled
            ),
            "content_below_asset": (
                asset_enabled
            ),
        },
        "requirements": (
            requirements
        ),
        "output_schema": (
            CreativeDirectionOutput
            .model_json_schema()
        ),
    }

    if diversity_constraints:
        payload["structural_diversity_plan"] = diversity_constraints
        requirements.append(
            "Keep locked structural fields exactly as supplied; choose at least two material "
            "changes from the listed unlocked fields relative to each accepted direction."
        )

    if correction_error:
        payload[
            "correction_required"
        ] = {
            "previous_validation_error": (
                correction_error
            ),
            "instruction": (
                "Return a corrected creative direction. "
                "Fix the exact validation problem while preserving "
                "the original event brief."
            ),
        }
        if diversity_constraints and diversity_constraints.get("failed_diagnostic"):
            retry_count = int(diversity_constraints.get("retry_count") or 0)
            payload["correction_required"]["diversity_overlap"] = diversity_constraints["failed_diagnostic"]
            payload["correction_required"]["instruction"] = (
                "Second retry: choose clearly unused values from the available unlocked alternatives. "
                "Change at least two unlocked major spatial choices relative to EACH accepted direction. "
                "Keep every locked structural field fixed. "
                "Preserve exactly the subject, event, theme, audience, palette, mood, important "
                "motifs and reference treatment. Do not invent scene-defining objects."
                if retry_count >= 2 else
                "First retry: change at least two unlocked major spatial choices relative to EACH accepted "
                "direction using the available alternatives. Keep every locked structural field fixed. "
                "Preserve exactly the subject, event, theme, audience, palette, mood, "
                "important motifs and reference treatment. Do not invent scene-defining objects."
            )

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


async def generate_creative_direction(
    direction_id: str,
    brief: dict,
    correction_error: str | None = None,
    previous_directions: list[str] | None = None,
    diversity_constraints: dict | None = None,
) -> CreativeDirectionOutput:
    model, role = (
        direction_configuration(
            direction_id
        )
    )

    logger.info(
        "Generating direction %s using model %s",
        direction_id.upper(),
        model,
    )

    prompt = (
        build_creative_direction_prompt(
            brief=brief,
            direction_id=(
                direction_id
            ),
            role=role,
            correction_error=(
                correction_error
            ),
            previous_directions=previous_directions,
            diversity_constraints=diversity_constraints,
        )
    )

    result = (
        await generate_text(
            model=model,
            prompt=prompt,
            response_format=(
                CreativeDirectionOutput
                .model_json_schema()
            ),
        )
    )

    direction = (
        CreativeDirectionOutput
        .model_validate_json(
            result[
                "response"
            ]
        )
    )

    direction.direction_id = (
        direction_id.upper()
    )

    direction.role = direction.role or "brief-led"

    direction = materialize_locked_structural_fields(
        direction, locked_structural_fields(brief)
    )

    direction.positive_prompt = (
        sanitize_positive_prompt(
            direction.positive_prompt,
            brief,
            direction.role,
        )
    )

    generated_negative = (
        direction.negative_prompt
        or ""
    )

    direction.negative_prompt = (
        deduplicate_negative_prompt(
            ", ".join(
                value
                for value
                in (
                    generated_negative.strip(
                        ", "
                    ),
                    (
                        "person, people, human, human figure, "
                        "man, woman, child, face, portrait, "
                        "silhouette, body, head, hands, arms, "
                        "legs, clothing, mannequin, character, "
                        "fictional character, hero, superhero, "
                        "masked character, mascot, costume, "
                        "costumed figure, humanoid"
                    ),
                    (
                        "text, typography, lettering, words, "
                        "letters, numbers, writing, calligraphy, "
                        "signature, logo, watermark, signage"
                    ),
                    palette_negative(brief),
                    background_negative_constraints(),
                )
                if value
            )
        )
    )

    #
    # LLM layout output is ignored.
    #
    # The user's Grill-Me placement answer is the source of truth.
    #
    direction.layout_guidance = (
        safe_layout(
            brief
        )
    )

    logger.info(
        "Completed direction %s using model %s",
        direction_id.upper(),
        model,
    )

    return (
        direction
    )
