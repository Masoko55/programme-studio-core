from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ProgrammeItem(BaseModel):
    time: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str | None = None


class GrillMeAnswers(BaseModel):
    """Happy Path 2 and 3 answers. All fields stay editable before generation."""

    event_type: str | None = None
    theme: str | None = None
    age_group: str | None = None
    primary_colour: str | None = None
    secondary_colour: str | None = None
    creative_description: str | None = None
    event_date: str | None = None
    start_time: str | None = None
    title_preference: str | None = None
    venue: str | None = None
    timezone: str | None = None
    programme: list[ProgrammeItem] | None = Field(default=None, max_length=15)
    output_language: str | None = None
    accessibility_preferences: list[str] | None = None
    headshot_shape: Literal["circle", "square", "rounded"] | None = None
    headshot_placement: Literal["left", "right"] | None = None
    logo_placement: Literal["left", "right"] | None = None
    headshot_consent_confirmed: bool | None = None
    logo_consent_confirmed: bool | None = None

    @field_validator("programme")
    @classmethod
    def max_fifteen_rows(cls, value):
        if value is not None and len(value) > 15:
            raise ValueError("A programme may contain at most 15 rows.")
        return value


class GrillMeSession(BaseModel):
    session_id: str
    status: Literal["questioning", "ready", "generating", "generated"]
    answers: GrillMeAnswers = Field(default_factory=GrillMeAnswers)
    assets: dict[str, dict] = Field(default_factory=dict)
    reference_number: str | None = None


class GenerateBackgroundsRequest(BaseModel):
    """Explicit confirmation that the reviewed Grill-Me answers may be frozen."""

    confirmed: bool

    @model_validator(mode="after")
    def confirm_generation(self):
        if not self.confirmed:
            raise ValueError("Set confirmed to true to generate backgrounds.")
        return self
