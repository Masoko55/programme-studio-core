import json
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
        "Where should the selected asset "
        "appear: left or right?"
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


def _save(
    session: GrillMeSession,
) -> GrillMeSession:
    directory = _directory(
        session.session_id
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

    return session


def load_session(
    session_id: str,
) -> GrillMeSession:
    path = _path(
        session_id
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
) -> str | None:
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
            value
            .strip()
        )

        if not cleaned:
            return True

        lowered = (
            cleaned
            .lower()
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

        if lowered in vague_values:
            return True

        if (
            field
            == "creative_description"
            and len(
                cleaned
            ) < 8
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

        if len(
            value
        ) == 0:
            return True

    return False


def _question(
    field: str,
    reason: str,
) -> GrillMeQuestion:
    return (
        GrillMeQuestion(
            field=field,
            question=(
                QUESTION_TEXT[
                    field
                ]
            ),
            reason=reason,
        )
    )


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

    for field in REQUIRED_FIELDS:
        value = (
            answers.get(
                field
            )
        )

        if _is_unclear(
            field,
            value,
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

    asset_type = (
        session.answers.asset_type
        or "none"
    )

    if (
        asset_type
        != "none"
    ):
        if not (
            session.answers
            .asset_placement
        ):
            questions.append(
                _question(
                    "asset_placement",
                    (
                        "An asset was selected "
                        "but no placement was provided."
                    ),
                )
            )

        if (
            asset_type
            == "headshot"
            and not (
                session.answers
                .headshot_shape
            )
        ):
            questions.append(
                _question(
                    "headshot_shape",
                    (
                        "A headshot was selected "
                        "but no shape was provided."
                    ),
                )
            )

        if not (
            session.answers
            .rights_and_consent_confirmed
        ):
            questions.append(
                _question(
                    (
                        "rights_and_consent_confirmed"
                    ),
                    (
                        "Rights and consent must "
                        "be confirmed before using "
                        "an uploaded asset."
                    ),
                )
            )

        expected_asset = (
            "headshot"
            if (
                asset_type
                == "headshot"
            )
            else "logo"
        )

        if (
            expected_asset
            not in session.assets
        ):
            questions.append(
                _question(
                    "asset_file",
                    (
                        f"The selected {expected_asset} "
                        "has not been uploaded yet."
                    ),
                )
            )

    return questions


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

    return _save(
        session
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
                (
                    "clear readable "
                    "programme text"
                ),
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

    return _refresh_status(
        session
    )


def apply_clarifications(
    session_id: str,
    values: dict,
) -> GrillMeSession:
    session = load_session(
        session_id
    )

    if session.status in {
        "generating",
        "generated",
    }:
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
        (
            "accessibility_preferences"
        ),
    }

    invalid = [
        key
        for key
        in values
        if (
            key not in allowed
            or key in protected
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
        (
            "clear readable "
            "programme text"
        ),
    ]

    session.answers = (
        GrillMeAnswers
        .model_validate(
            current
        )
    )

    return _refresh_status(
        session
    )


async def store_selected_asset(
    session_id: str,
    upload: UploadFile,
) -> GrillMeSession:
    session = load_session(
        session_id
    )

    if session.status in {
        "generating",
        "generated",
    }:
        raise ValueError(
            "This Grill-Me session is frozen "
            "after generation."
        )

    asset_type = (
        session.answers.asset_type
    )

    if asset_type not in {
        "headshot",
        "logo",
    }:
        raise ValueError(
            "The form did not select "
            "an image or logo."
        )

    suffix = (
        ALLOWED_ASSET_TYPES
        .get(
            upload.content_type
            or ""
        )
    )

    if suffix is None:
        raise ValueError(
            "Assets must be JPEG or PNG files."
        )

    data = await upload.read(
        MAX_ASSET_BYTES
        + 1
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
            existing_path = (
                Path(
                    existing[
                        "path"
                    ]
                )
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

    return _refresh_status(
        session
    )


def freeze_for_generation(
    session_id: str,
    reference_number: str,
) -> tuple[
    GrillMeSession,
    dict,
    dict,
]:
    session = load_session(
        session_id
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
    }

    asset_type = (
        answer[
            "asset_type"
        ]
    )

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
        "asset_type": (
            asset_type
        ),
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
            answer.get(
                "asset_placement"
            )
            if (
                asset_type
                == "headshot"
            )
            else None
        ),
        "logo_placement": (
            answer.get(
                "asset_placement"
            )
            if (
                asset_type
                == "logo"
            )
            else None
        ),
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
    session = load_session(
        session_id
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
    path = (
        settings.programme_data_path
        / reference_number
        / "pending-final.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            details,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path