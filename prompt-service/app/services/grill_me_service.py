import json
import re
import shutil
import uuid

from pathlib import Path

from fastapi import UploadFile

from app.config.settings import (
    settings,
)

from app.schemas.grill_me import (
    GrillMeAnswers,
    GrillMeForm,
    GrillMeQuestion,
    GrillMeSession,
)

from app.services.grill_me_semantic import (
    review_creative_context,
)


MAX_ASSET_BYTES = (
    10
    * 1024
    * 1024
)


ALLOWED_ASSET_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


REQUIRED_FIELDS = (
    "event_type",
    "theme",
    "age_group",
    "primary_colour",
    "secondary_colour",
    "creative_description",
    "event_date",
    "start_time",
    "title_preference",
    "venue",
    "programme",
    "output_language",
)


COLOUR_MODIFIERS = {
    "neon",
    "fluorescent",
    "electric",
    "pastel",
    "bright",
    "vibrant",
    "muted",
    "soft",
    "dark",
    "deep",
    "light",
    "metallic",
    "glowing",
    "glow",
    "luminous",
    "radiant",
}


KNOWN_COLOUR_TERMS = {
    "black",
    "white",
    "grey",
    "gray",
    "silver",
    "charcoal",
    "red",
    "orange",
    "yellow",
    "lime",
    "lime green",
    "green",
    "forest green",
    "mint green",
    "teal",
    "turquoise",
    "cyan",
    "aqua",
    "blue",
    "sky blue",
    "royal blue",
    "navy",
    "navy blue",
    "purple",
    "violet",
    "lavender",
    "lilac",
    "magenta",
    "fuchsia",
    "pink",
    "hot pink",
    "rose",
    "rose pink",
    "coral",
    "peach",
    "maroon",
    "burgundy",
    "brown",
    "beige",
    "cream",
    "ivory",
    "champagne",
    "gold",
    "golden",
    "rose gold",
}


QUESTION_TEXT = {
    "event_type": (
        "What type of event is this?"
    ),
    "theme": (
        "What theme should the event follow?"
    ),
    "age_group": (
        "Who is the event intended for?"
    ),
    "primary_colour": (
        "What should the primary colour be?"
    ),
    "secondary_colour": (
        "What should the secondary colour be?"
    ),
    "creative_description": (
        "Please describe the visual mood "
        "or creative direction more clearly."
    ),
    "background_inspiration": (
        "What would you like people to see in the background? "
        "For example, a city skyline, flowers, balloons, a stage, "
        "a pattern, or something abstract."
    ),
    "theme_reference_treatment": (
        "Which visual traits from the reference "
        "should influence the background?"
    ),
    "event_date": (
        "What is the event date?"
    ),
    "start_time": (
        "What time does the event start?"
    ),
    "title_preference": (
        "What title should appear "
        "on the final programme?"
    ),
    "venue": (
        "Where is the venue?"
    ),
    "programme": (
        "Please provide at least one "
        "programme row."
    ),
    "output_language": (
        "What language should be used "
        "for the final programme?"
    ),
    "asset_placement": (
        "Where should the selected asset be placed at the top "
        "of the programme: left, center, or right? "
        "All title, event details and programme text will appear below it."
    ),
    "headshot_shape": (
        "What shape should the headshot use: "
        "circle, square or rounded?"
    ),
    "rights_and_consent_confirmed": (
        "Please confirm that you have rights "
        "and consent to use the selected asset."
    ),
    "asset_file": (
        "Please upload the selected asset "
        "before generation."
    ),
}


def _directory(
    session_id: str,
) -> Path:
    return (
        settings.programme_data_path
        / "intake-sessions"
        / session_id
    )


def _path(
    session_id: str,
) -> Path:
    return (
        _directory(
            session_id
        )
        / "session.json"
    )


def _archive_session_directory(
    session_id: str,
) -> Path:
    return (
        settings.prompt_archive_path
        / "intake-sessions"
        / session_id
    )


def _archive_session_path(
    session_id: str,
) -> Path:
    return (
        _archive_session_directory(
            session_id
        )
        / "session.json"
    )


def _mirror_session(
    session: GrillMeSession,
) -> None:
    destination = (
        _archive_session_path(
            session.session_id
        )
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        session.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    for _, asset in (
        session.assets.items()
    ):
        source_value = (
            asset.get(
                "path"
            )
        )

        if not source_value:
            continue

        source = Path(
            source_value
        )

        if not source.is_file():
            continue

        archive_assets = (
            destination.parent
            / "assets"
        )

        archive_assets.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            archive_assets
            / source.name,
        )


def _save(
    session: GrillMeSession,
) -> GrillMeSession:
    directory = (
        _directory(
            session.session_id
        )
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _path(
        session.session_id
    ).write_text(
        session.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    _mirror_session(
        session
    )

    return session


def load_session(
    session_id: str,
) -> GrillMeSession:
    path = (
        _path(
            session_id
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            "Grill-Me session not found: "
            f"{session_id}"
        )

    return (
        GrillMeSession
        .model_validate_json(
            path.read_text(
                encoding="utf-8"
            )
        )
    )


def _clean_string(
    value,
):
    if not isinstance(
        value,
        str,
    ):
        return value

    cleaned = (
        value.strip()
    )

    return (
        cleaned
        if cleaned
        else None
    )


def _normalize_colour_text(
    value: str | None,
) -> str | None:
    if not value:
        return None

    normalized = (
        value.strip()
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
        normalized
        if normalized
        else None
    )


def _is_hex_colour(
    value: str,
) -> bool:
    return bool(
        re.fullmatch(
            r"#?[0-9a-fA-F]{6}",
            value,
        )
    )


def _colour_contains_known_hue(
    value: str,
) -> bool:
    if (
        _is_hex_colour(
            value
        )
    ):
        return True

    candidates = sorted(
        KNOWN_COLOUR_TERMS,
        key=len,
        reverse=True,
    )

    for colour in candidates:
        if re.search(
            rf"\b{re.escape(colour)}\b",
            value,
        ):
            return True

    return False


def _colour_modifier_only(
    value: str | None,
) -> bool:
    normalized = (
        _normalize_colour_text(
            value
        )
    )

    if not normalized:
        return False

    if (
        _is_hex_colour(
            normalized
        )
    ):
        return False

    if (
        _colour_contains_known_hue(
            normalized
        )
    ):
        return False

    words = set(
        normalized.split()
    )

    return bool(
        words
        and words.issubset(
            COLOUR_MODIFIERS
        )
    )


def _colour_is_unknown(
    value: str | None,
) -> bool:
    normalized = (
        _normalize_colour_text(
            value
        )
    )

    if not normalized:
        return False

    if (
        _is_hex_colour(
            normalized
        )
    ):
        return False

    if (
        _colour_contains_known_hue(
            normalized
        )
    ):
        return False

    if (
        _colour_modifier_only(
            normalized
        )
    ):
        return False

    return True


def _is_unclear(
    field: str,
    value,
) -> bool:
    if value is None:
        return True

    if isinstance(
        value,
        str,
    ):
        cleaned = (
            value.strip()
        )

        if not cleaned:
            return True

        lowered = (
            cleaned.lower()
        )

        vague_values = {
            "idk",
            "i don't know",
            "dont know",
            "not sure",
            "unsure",
            "whatever",
            "anything",
            "something",
            "n/a",
            "na",
            "unknown",
            "tbd",
            "later",
        }

        if (
            lowered
            in vague_values
        ):
            return True

        if (
            field
            == "creative_description"
            and len(
                cleaned
            )
            < 8
        ):
            return True

    if (
        field
        == "programme"
    ):
        if not isinstance(
            value,
            list,
        ):
            return True

        if (
            len(
                value
            )
            == 0
        ):
            return True

    return False


def _question(
    field: str,
    reason: str,
    question: str | None = None,
) -> GrillMeQuestion:
    return (
        GrillMeQuestion(
            field=field,
            question=(
                question
                or QUESTION_TEXT[
                    field
                ]
            ),
            reason=reason,
        )
    )


def _colour_question(
    field: str,
    value: str | None,
) -> GrillMeQuestion | None:
    if (
        _colour_modifier_only(
            value
        )
    ):
        normalized = (
            _normalize_colour_text(
                value
            )
            or ""
        )

        return (
            _question(
                field,
                (
                    "A colour style was provided "
                    "without a measurable base hue."
                ),
                (
                    f"Which {normalized} colour should "
                    f"the {field.replace('_', ' ')} be? "
                    "For example neon pink, neon blue, "
                    "neon purple or neon green."
                ),
            )
        )

    if (
        _colour_is_unknown(
            value
        )
    ):
        normalized = (
            _normalize_colour_text(
                value
            )
            or str(
                value
            )
        )

        return (
            _question(
                field,
                (
                    "The colour could not be mapped "
                    "to a known colour family."
                ),
                (
                    f"What base colour should '{normalized}' "
                    "belong to? You can also provide a "
                    "six-digit hex colour such as #FF00FF."
                ),
            )
        )

    return None


def _semantic_questions(
    session: GrillMeSession,
) -> list[
    GrillMeQuestion
]:
    items = (
        session.creative_context.get(
            "clarification_questions",
            [],
        )
        if session.creative_context
        else []
    )

    questions = []

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        field = (
            str(
                item.get(
                    "field"
                )
                or ""
            )
            .strip()
        )

        question = (
            str(
                item.get(
                    "question"
                )
                or ""
            )
            .strip()
        )

        reason = (
            str(
                item.get(
                    "reason"
                )
                or ""
            )
            .strip()
        )

        if not (
            field
            and question
        ):
            continue

        questions.append(
            GrillMeQuestion(
                field=field,
                question=question,
                reason=(
                    reason
                    or (
                        "The creative intent "
                        "needs clarification."
                    )
                ),
            )
        )

    return questions


def _asset_placement_question(
    asset_type: str,
) -> GrillMeQuestion:
    asset_label = (
        "headshot"
        if asset_type
        == "headshot"
        else "logo"
    )

    return (
        _question(
            "asset_placement",
            (
                f"A {asset_label} is being used, so Grill-Me must "
                "clarify its horizontal position before layout generation. "
                "The asset always sits at the top and all programme text "
                "flows underneath it."
            ),
            (
                f"Where should the {asset_label} be placed at the top "
                "of the programme: left, center, or right? "
                "All title, event details and programme text will appear "
                "below the asset."
            ),
        )
    )


def _asset_questions(session: GrillMeSession, existing_fields: set[str]) -> list[GrillMeQuestion]:
    questions = []
    asset_type = (
        session.answers.asset_type
        or "none"
    )

    #
    # Asset-specific questions only exist when the user
    # actually selected a headshot or logo.
    #
    if (
        asset_type
        != "none"
    ):
        #
        # Placement MUST be answered through Grill-Me.
        #
        # Vertical position is fixed:
        #     TOP
        #
        # Grill-Me only asks:
        #     left / center / right
        #
        # All text comes underneath the asset.
        #
        if (
            not session.answers.asset_placement
            and "asset_placement" not in existing_fields
        ):
            questions.append(_asset_placement_question(asset_type))
            existing_fields.add("asset_placement")

        if (
            asset_type == "headshot"
            and not session.answers.headshot_shape
            and "headshot_shape" not in existing_fields
        ):
            questions.append(_question(
                "headshot_shape",
                "A headshot was selected but no shape was provided.",
            ))
            existing_fields.add("headshot_shape")

        if (
            not session.answers.rights_and_consent_confirmed
            and "rights_and_consent_confirmed" not in existing_fields
        ):
            questions.append(_question(
                "rights_and_consent_confirmed",
                "Rights and consent must be confirmed before using an uploaded asset.",
            ))
            existing_fields.add("rights_and_consent_confirmed")

        expected_asset = (
            "headshot"
            if (
                asset_type
                == "headshot"
            )
            else "logo"
        )

        if (
            expected_asset not in session.assets
            and "asset_file" not in existing_fields
        ):
            questions.append(_question(
                "asset_file",
                f"The selected {expected_asset} has not been uploaded yet.",
            ))
            existing_fields.add("asset_file")

    return questions


def _append_semantic_questions(
    session: GrillMeSession,
    questions: list[GrillMeQuestion],
    existing_fields: set[str],
) -> None:
    for item in _semantic_questions(session):
        if item.field in existing_fields:
            continue
        questions.append(item)
        existing_fields.add(item.field)


def _append_colour_questions(
    answers: dict,
    questions: list[GrillMeQuestion],
    existing_fields: set[str],
) -> None:
    for colour_field in ("primary_colour", "secondary_colour"):
        if colour_field in existing_fields:
            continue
        question = _colour_question(colour_field, answers.get(colour_field))
        if question is not None:
            questions.append(question)
            existing_fields.add(colour_field)


def evaluate_session(
    session: GrillMeSession,
) -> list[
    GrillMeQuestion
]:
    answers = (
        session.answers
        .model_dump()
    )

    questions = []

    for field in (
        REQUIRED_FIELDS
    ):
        value = (
            answers.get(
                field
            )
        )

        if (
            _is_unclear(
                field,
                value,
            )
        ):
            questions.append(
                _question(
                    field,
                    (
                        "This information is "
                        "missing or unclear."
                    ),
                )
            )

    existing_fields = {
        item.field
        for item
        in questions
    }

    _append_colour_questions(answers, questions, existing_fields)

    _append_semantic_questions(session, questions, existing_fields)

    questions.extend(_asset_questions(session, existing_fields))

    return questions


def _run_semantic_review(
    session: GrillMeSession,
) -> None:
    session.creative_context = (
        review_creative_context(
            session.answers.model_dump(
                mode="json"
            )
        )
    )


def _refresh_status(
    session: GrillMeSession,
) -> GrillMeSession:
    pending = (
        evaluate_session(
            session
        )
    )

    session.clarification_questions = (
        pending
    )

    if pending:
        session.status = (
            "clarification_required"
        )

    else:
        session.status = (
            "ready"
        )

    return (
        _save(
            session
        )
    )


def create_session_from_form(
    form: GrillMeForm,
) -> GrillMeSession:
    payload = (
        form.model_dump()
    )

    answers = (
        GrillMeAnswers(
            **payload,
            timezone=(
                settings.user_timezone
            ),
            accessibility_preferences=[
                "high contrast",
                "clear readable programme text",
            ],
        )
    )

    session = (
        GrillMeSession(
            session_id=(
                uuid.uuid4().hex
            ),
            status="reviewing",
            answers=answers,
        )
    )

    _run_semantic_review(
        session
    )

    return (
        _refresh_status(
            session
        )
    )


def apply_clarifications(
    session_id: str,
    values: dict,
) -> GrillMeSession:
    session = (
        load_session(
            session_id
        )
    )

    if (
        session.status
        in {
            "generating",
            "generated",
        }
    ):
        raise ValueError(
            "This Grill-Me session is frozen "
            "after generation."
        )

    allowed = set(
        GrillMeAnswers
        .model_fields
        .keys()
    )

    protected = {
        "timezone",
        "accessibility_preferences",
    }

    invalid = [
        key
        for key
        in values
        if (
            key
            not in allowed
            or key
            in protected
        )
    ]

    if invalid:
        raise ValueError(
            "Unsupported clarification fields: "
            + ", ".join(
                invalid
            )
        )

    current = (
        session.answers
        .model_dump()
    )

    for key, value in (
        values.items()
    ):
        current[
            key
        ] = (
            _clean_string(
                value
            )
        )

    current[
        "timezone"
    ] = (
        session.answers.timezone
        or settings.user_timezone
    )

    current[
        "accessibility_preferences"
    ] = [
        "high contrast",
        "clear readable programme text",
    ]

    session.answers = (
        GrillMeAnswers
        .model_validate(
            current
        )
    )

    _run_semantic_review(
        session
    )

    return (
        _refresh_status(
            session
        )
    )


async def store_selected_asset(
    session_id: str,
    upload: UploadFile,
) -> GrillMeSession:
    session = (
        load_session(
            session_id
        )
    )

    if (
        session.status
        in {
            "generating",
            "generated",
        }
    ):
        raise ValueError(
            "This Grill-Me session is frozen "
            "after generation."
        )

    asset_type = (
        session.answers
        .asset_type
    )

    if (
        asset_type
        not in {
            "headshot",
            "logo",
        }
    ):
        raise ValueError(
            "The form did not select "
            "a headshot or logo."
        )

    suffix = (
        ALLOWED_ASSET_TYPES
        .get(
            upload.content_type
            or ""
        )
    )

    if (
        suffix
        is None
    ):
        raise ValueError(
            "Assets must be JPEG or PNG files."
        )

    data = (
        await upload.read(
            MAX_ASSET_BYTES
            + 1
        )
    )

    if (
        len(
            data
        )
        > MAX_ASSET_BYTES
    ):
        raise ValueError(
            "Assets must be no larger "
            "than 10 MB."
        )

    assets_directory = (
        _directory(
            session_id
        )
        / "assets"
    )

    assets_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for existing_name in (
        "headshot",
        "logo",
    ):
        existing = (
            session.assets
            .get(
                existing_name
            )
        )

        if existing:
            existing_path = Path(
                existing[
                    "path"
                ]
            )

            if (
                existing_path
                .exists()
            ):
                existing_path.unlink()

    asset_path = (
        assets_directory
        / (
            f"{asset_type}"
            f"{suffix}"
        )
    )

    asset_path.write_bytes(
        data
    )

    session.assets = {
        asset_type: {
            "path": str(
                asset_path
            ),
            "content_type": (
                upload.content_type
            ),
            "original_filename": (
                upload.filename
            ),
        }
    }

    return (
        _refresh_status(
            session
        )
    )


def freeze_for_generation(
    session_id: str,
    reference_number: str,
) -> tuple[
    GrillMeSession,
    dict,
    dict,
]:
    session = (
        load_session(
            session_id
        )
    )

    pending = (
        evaluate_session(
            session
        )
    )

    if pending:
        fields = [
            item.field
            for item
            in pending
        ]

        raise ValueError(
            "GRILL_ME_INCOMPLETE:"
            + ",".join(
                fields
            )
        )

    if (
        session.status
        == "generated"
    ):
        raise ValueError(
            "This Grill-Me session has "
            "already generated backgrounds."
        )

    if not (
        session.answers.timezone
    ):
        session.answers.timezone = (
            settings.user_timezone
        )

    session.status = (
        "generating"
    )

    session.reference_number = (
        reference_number
    )

    _save(
        session
    )

    answer = (
        session.answers
        .model_dump()
    )

    asset_type = (
        answer[
            "asset_type"
        ]
    )

    asset_placement = (
        answer.get(
            "asset_placement"
        )
        if (
            asset_type
            != "none"
        )
        else None
    )

    #
    # The vertical position is deliberately fixed.
    #
    # The user only chooses horizontal placement:
    #
    #     left
    #     center
    #     right
    #
    # Every text block starts underneath this top asset area.
    #
    asset_vertical_position = (
        "top"
        if (
            asset_type
            != "none"
        )
        else None
    )

    text_flow = (
        "below_asset"
        if (
            asset_type
            != "none"
        )
        else "standard"
    )

    background_brief = {
        "event_type": (
            answer[
                "event_type"
            ]
        ),
        "theme": (
            answer[
                "theme"
            ]
        ),
        "age_group": (
            answer[
                "age_group"
            ]
        ),
        "primary_colour": (
            answer[
                "primary_colour"
            ]
        ),
        "secondary_colour": (
            answer[
                "secondary_colour"
            ]
        ),
        "creative_description": (
            answer[
                "creative_description"
            ]
        ),
        "background_inspiration": (
            answer.get(
                "background_inspiration"
            )
        ),
        "theme_reference_treatment": (
            answer.get(
                "theme_reference_treatment"
            )
        ),
        "grill_me_context": (
            session.creative_context
        ),
        "title_preference": (
            answer[
                "title_preference"
            ]
        ),
        "event_date": (
            answer[
                "event_date"
            ]
        ),
        "start_time": (
            answer[
                "start_time"
            ]
        ),
        "timezone": (
            answer[
                "timezone"
            ]
        ),
        "output_language": (
            answer[
                "output_language"
            ]
        ),
        "accessibility_preferences": (
            answer[
                "accessibility_preferences"
            ]
        ),

        #
        # Asset layout is also given to the
        # creative-direction/background agents.
        #
        "asset_type": (
            asset_type
        ),
        "asset_placement": (
            asset_placement
        ),
        "asset_vertical_position": (
            asset_vertical_position
        ),
        "text_flow": (
            text_flow
        ),
    }

    asset = (
        session.assets
        .get(
            asset_type,
            {},
        )
        if (
            asset_type
            != "none"
        )
        else {}
    )

    asset_path = (
        asset.get(
            "path"
        )
    )

    final_details = {
        "session_id": (
            session.session_id
        ),
        "reference_number": (
            reference_number
        ),
        "event_type": (
            answer[
                "event_type"
            ]
        ),
        "theme": (
            answer[
                "theme"
            ]
        ),
        "age_group": (
            answer[
                "age_group"
            ]
        ),
        "primary_colour": (
            answer[
                "primary_colour"
            ]
        ),
        "secondary_colour": (
            answer[
                "secondary_colour"
            ]
        ),
        "creative_description": (
            answer[
                "creative_description"
            ]
        ),
        "background_inspiration": (
            answer.get(
                "background_inspiration"
            )
        ),
        "theme_reference_treatment": (
            answer.get(
                "theme_reference_treatment"
            )
        ),
        "grill_me_context": (
            session.creative_context
        ),
        "title": (
            answer[
                "title_preference"
            ]
        ),
        "event_date": (
            answer[
                "event_date"
            ]
        ),
        "start_time": (
            answer[
                "start_time"
            ]
        ),
        "timezone": (
            answer[
                "timezone"
            ]
        ),
        "venue": (
            answer[
                "venue"
            ]
        ),
        "programme": (
            answer[
                "programme"
            ]
        ),
        "output_language": (
            answer[
                "output_language"
            ]
        ),
        "accessibility_preferences": (
            answer[
                "accessibility_preferences"
            ]
        ),

        #
        # Generic asset information.
        #
        "asset_type": (
            asset_type
        ),
        "asset_placement": (
            asset_placement
        ),
        "asset_vertical_position": (
            asset_vertical_position
        ),
        "text_flow": (
            text_flow
        ),

        #
        # Asset paths.
        #
        "headshot_path": (
            asset_path
            if (
                asset_type
                == "headshot"
            )
            else None
        ),
        "logo_path": (
            asset_path
            if (
                asset_type
                == "logo"
            )
            else None
        ),

        #
        # Headshot-only options.
        #
        "headshot_shape": (
            answer.get(
                "headshot_shape"
            )
            if (
                asset_type
                == "headshot"
            )
            else None
        ),
        "headshot_placement": (
            asset_placement
            if (
                asset_type
                == "headshot"
            )
            else None
        ),

        #
        # Logo-only placement.
        #
        "logo_placement": (
            asset_placement
            if (
                asset_type
                == "logo"
            )
            else None
        ),

        #
        # Explicit downstream layout contract.
        #
        "layout_contract": {
            "asset": {
                "enabled": (
                    asset_type
                    != "none"
                ),
                "type": (
                    asset_type
                ),
                "vertical_position": (
                    asset_vertical_position
                ),
                "horizontal_position": (
                    asset_placement
                ),
            },
            "text": {
                "flow": (
                    text_flow
                ),
                "must_begin_below_asset": (
                    asset_type
                    != "none"
                ),
            },
        },

        "rights_and_consent_confirmed": (
            bool(
                answer.get(
                    "rights_and_consent_confirmed"
                )
            )
            if (
                asset_type
                != "none"
            )
            else True
        ),
    }

    return (
        session,
        background_brief,
        final_details,
    )


def mark_generated(
    session_id: str,
) -> None:
    session = (
        load_session(
            session_id
        )
    )

    session.status = (
        "generated"
    )

    _save(
        session
    )


def persist_pending_final(
    reference_number: str,
    details: dict,
) -> Path:
    primary_path = (
        settings.programme_data_path
        / reference_number
        / "pending-final.json"
    )

    primary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = json.dumps(
        details,
        ensure_ascii=False,
        indent=2,
    )

    primary_path.write_text(
        payload,
        encoding="utf-8",
    )

    archive_path = (
        settings.prompt_archive_path
        / reference_number
        / "pending-final.json"
    )

    archive_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_path.write_text(
        payload,
        encoding="utf-8",
    )

    return primary_path
