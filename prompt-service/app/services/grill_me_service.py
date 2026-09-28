"""Persisted Grill-Me questionnaire.

Grill-Me gathers the event brief before creative directions are created.

Timezone is system/session metadata. Grill-Me does not ask the user for it.
The configured deployment timezone is inserted automatically when a session
is created.

Programme text is held back until one background is selected.
"""

import json
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config.settings import settings
from app.schemas.grill_me import (
    GrillMeAnswers,
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


# timezone deliberately excluded
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

def create_session() -> GrillMeSession:
    session = GrillMeSession(
        session_id=uuid.uuid4().hex,
        status="questioning",
        answers=GrillMeAnswers(
            timezone=settings.user_timezone,
            accessibility_preferences=[
                "high contrast",
                "clear readable programme text",
            ],
        ),
    )

    return _save(
        session
    )




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

def questions(
    session: GrillMeSession,
) -> list[dict]:
    answers = (
        session.answers
        .model_dump()
    )

    prompts = {
        "event_type": (
            "What type of event are you creating?"
        ),

        "theme": (
            "What is the event theme?"
        ),

        "age_group": (
            "Who is the audience or age group?"
        ),

        "primary_colour": (
            "What is the primary suggested colour?"
        ),

        "secondary_colour": (
            "What is the secondary suggested colour?"
        ),

        "creative_description": (
            "Describe the visual mood and creative direction."
        ),

        "event_date": (
            "What is the event date?"
        ),

        "start_time": (
            "What is the start time?"
        ),

        "title_preference": (
            "What title should appear on the final programme?"
        ),

        "venue": (
            "Where is the venue?"
        ),

        "programme": (
            "Add up to 15 programme rows."
        ),

        "output_language": (
            "What output language should be used?"
        ),
    }

    result = [
        {
            "field": field,
            "question": prompts[
                field
            ],
        }
        for field
        in REQUIRED_FIELDS
        if (
            answers.get(
                field
            )
            in (
                None,
                "",
            )
        )
    ]

    if (
        "headshot"
        in session.assets
        and not answers.get(
            "headshot_consent_confirmed"
        )
    ):
        result.append(
            {
                "field": (
                    "headshot_consent_confirmed"
                ),
                "question": (
                    "Confirm you have rights and "
                    "consent to use the headshot."
                ),
            }
        )

    if (
        "headshot"
        in session.assets
        and not answers.get(
            "headshot_shape"
        )
    ):
        result.append(
            {
                "field": (
                    "headshot_shape"
                ),
                "question": (
                    "Choose the headshot shape: "
                    "circle, square, or rounded."
                ),
            }
        )

    if (
        "headshot"
        in session.assets
        and not answers.get(
            "headshot_placement"
        )
    ):
        result.append(
            {
                "field": (
                    "headshot_placement"
                ),
                "question": (
                    "Choose the headshot placement: "
                    "left or right."
                ),
            }
        )

    if (
        "logo"
        in session.assets
        and not answers.get(
            "logo_consent_confirmed"
        )
    ):
        result.append(
            {
                "field": (
                    "logo_consent_confirmed"
                ),
                "question": (
                    "Confirm you have rights and "
                    "consent to use the logo."
                ),
            }
        )

    if (
        "logo"
        in session.assets
        and not answers.get(
            "logo_placement"
        )
    ):
        result.append(
            {
                "field": (
                    "logo_placement"
                ),
                "question": (
                    "Choose the logo placement: "
                    "left or right."
                ),
            }
        )

    return result




def update_answers(
    session_id: str,
    patch: GrillMeAnswers,
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

    values = (
        patch.model_dump(
            exclude_unset=True
        )
    )

    # timezone is system-managed
    values.pop(
        "timezone",
        None,
    )

    merged = {
        **session.answers.model_dump(),
        **values,
        "timezone": (
            session.answers.timezone
            or settings.user_timezone
        ),
    }

    session.answers = (
        GrillMeAnswers
        .model_validate(
            merged
        )
    )

    session.status = (
        "ready"
        if not questions(
            session
        )
        else "questioning"
    )

    return _save(
        session
    )


def append_programme_item(
    session_id: str,
    item,
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

    rows = [
        row.model_dump()
        for row
        in (
            session.answers.programme
            or []
        )
    ]

    if len(
        rows
    ) >= 15:
        raise ValueError(
            "A programme may contain "
            "at most 15 rows."
        )

    rows.append(
        item.model_dump()
    )

    session.answers = (
        GrillMeAnswers
        .model_validate(
            {
                **session.answers.model_dump(),
                "programme": rows,
            }
        )
    )

    session.status = (
        "ready"
        if not questions(
            session
        )
        else "questioning"
    )

    return _save(
        session
    )


async def store_asset(
    session_id: str,
    asset_name: str,
    upload: UploadFile,
) -> GrillMeSession:
    if asset_name not in {
        "headshot",
        "logo",
    }:
        raise ValueError(
            "asset_name must be headshot or logo."
        )

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

    if len(
        data
    ) > MAX_ASSET_BYTES:
        raise ValueError(
            "Assets must be no larger "
            "than 10 MB."
        )

    asset_path = (
        _directory(
            session_id
        )
        / "assets"
        / f"{asset_name}{suffix}"
    )

    asset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    asset_path.write_bytes(
        data
    )

    session.assets[
        asset_name
    ] = {
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

    session.status = (
        "ready"
        if not questions(
            session
        )
        else "questioning"
    )

    return _save(
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

    missing = questions(
        session
    )

    if missing:
        fields = [
            item[
                "field"
            ]
            for item
            in missing
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
            "This Grill-Me session has already "
            "generated backgrounds."
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

        # retained internally for temporal context
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

        # retained in data, not rendered on poster
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

        "headshot_path": (
            session.assets
            .get(
                "headshot",
                {},
            )
            .get(
                "path"
            )
        ),

        "logo_path": (
            session.assets
            .get(
                "logo",
                {},
            )
            .get(
                "path"
            )
        ),

        "headshot_shape": (
            answer.get(
                "headshot_shape"
            )
            or "rounded"
        ),

        "headshot_placement": (
            answer.get(
                "headshot_placement"
            )
            or "left"
        ),

        "logo_placement": (
            answer.get(
                "logo_placement"
            )
            or "right"
        ),

        "rights_and_consent_confirmed": bool(
            (
                "headshot"
                not in session.assets
                or answer.get(
                    "headshot_consent_confirmed"
                )
            )
            and
            (
                "logo"
                not in session.assets
                or answer.get(
                    "logo_consent_confirmed"
                )
            )
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