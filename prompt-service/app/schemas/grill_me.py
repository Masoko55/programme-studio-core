from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class ProgrammeItem(
    BaseModel
):
    time: str = Field(
        min_length=1
    )

    title: str = Field(
        min_length=1
    )

    description: str | None = None


class GrillMeForm(
    BaseModel
):
    event_type: str | None = None
    theme: str | None = None
    age_group: str | None = None

    primary_colour: str | None = None
    secondary_colour: str | None = None

    creative_description: str | None = None

    background_subject: str | None = None
    background_style: str | None = None
    background_motifs: str | None = None
    background_exclusions: str | None = None
    background_composition: str | None = None

    theme_reference_treatment: (
        str | None
    ) = None

    event_date: str | None = None
    start_time: str | None = None

    title_preference: str | None = None
    venue: str | None = None

    programme: list[
        ProgrammeItem
    ] | None = Field(
        default=None,
        max_length=15,
    )

    output_language: str | None = "en"

    asset_type: Literal[
        "none",
        "headshot",
        "logo",
    ] = "none"

    asset_placement: Literal[
        "left",
        "right",
    ] | None = None

    headshot_shape: Literal[
        "circle",
        "square",
        "rounded",
    ] | None = None

    rights_and_consent_confirmed: (
        bool | None
    ) = None

    @field_validator(
        "programme"
    )
    @classmethod
    def validate_programme(
        cls,
        value,
    ):
        if (
            value is not None
            and len(
                value
            ) > 15
        ):
            raise ValueError(
                "A programme may contain "
                "at most 15 rows."
            )

        return value

    @model_validator(
        mode="after"
    )
    def validate_asset_options(
        self,
    ):
        if (
            self.asset_type
            == "none"
        ):
            self.asset_placement = None
            self.headshot_shape = None
            self.rights_and_consent_confirmed = None

            return self

        if (
            self.asset_type
            == "logo"
        ):
            self.headshot_shape = None

        return self


class GrillMeClarificationRequest(
    BaseModel
):
    answers: dict[
        str,
        object,
    ] = Field(
        default_factory=dict
    )


class GrillMeAnswers(
    BaseModel
):
    event_type: str | None = None
    theme: str | None = None
    age_group: str | None = None

    primary_colour: str | None = None
    secondary_colour: str | None = None

    creative_description: str | None = None

    background_subject: str | None = None
    background_style: str | None = None
    background_motifs: str | None = None
    background_exclusions: str | None = None
    background_composition: str | None = None

    theme_reference_treatment: (
        str | None
    ) = None

    event_date: str | None = None
    start_time: str | None = None

    title_preference: str | None = None
    venue: str | None = None

    timezone: str | None = None

    programme: list[
        ProgrammeItem
    ] | None = Field(
        default=None,
        max_length=15,
    )

    output_language: str | None = "en"

    accessibility_preferences: (
        list[str] | None
    ) = None

    asset_type: Literal[
        "none",
        "headshot",
        "logo",
    ] = "none"

    asset_placement: Literal[
        "left",
        "right",
    ] | None = None

    headshot_shape: Literal[
        "circle",
        "square",
        "rounded",
    ] | None = None

    rights_and_consent_confirmed: (
        bool | None
    ) = None

    @field_validator(
        "programme"
    )
    @classmethod
    def max_fifteen_rows(
        cls,
        value,
    ):
        if (
            value is not None
            and len(
                value
            ) > 15
        ):
            raise ValueError(
                "A programme may contain "
                "at most 15 rows."
            )

        return value


class GrillMeQuestion(
    BaseModel
):
    field: str
    question: str
    reason: str


class GrillMeSession(
    BaseModel
):
    session_id: str

    status: Literal[
        "reviewing",
        "clarification_required",
        "ready",
        "generating",
        "generated",
    ]

    answers: GrillMeAnswers = Field(
        default_factory=(
            GrillMeAnswers
        )
    )

    clarification_questions: list[
        GrillMeQuestion
    ] = Field(
        default_factory=list
    )

    creative_context: dict = Field(
        default_factory=dict
    )

    assets: dict[
        str,
        dict,
    ] = Field(
        default_factory=dict
    )

    reference_number: str | None = None


class GenerateBackgroundsRequest(
    BaseModel
):
    confirmed: bool

    @model_validator(
        mode="after"
    )
    def confirm_generation(
        self,
    ):
        if not self.confirmed:
            raise ValueError(
                "Set confirmed to true "
                "to generate backgrounds."
            )

        return self