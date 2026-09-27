from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ProgrammeItem(BaseModel):
    time: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: Optional[str] = None


class FinalProgrammeDetails(BaseModel):
    """Event details supplied only after a background has been selected."""

    title: str = Field(min_length=1)
    event_date: str = Field(min_length=1)
    start_time: str = Field(min_length=1)
    timezone: str = Field(min_length=1)
    venue: Optional[str] = None
    programme: list[ProgrammeItem] = Field(default_factory=list, max_length=15)
    output_language: str = Field(default="en", min_length=1)
    accessibility_preferences: list[str] = Field(default_factory=list)


class FinalSelectionRequest(BaseModel):
    """Client-selected candidate and the details that turn it into the final."""

    engine_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    direction_id: str = Field(pattern=r"^[ABC]$")
    programme_details: FinalProgrammeDetails | None = None
    headshot_path: Optional[str] = None
    logo_path: Optional[str] = None
    headshot_shape: Optional[str] = Field(default=None, pattern=r"^(circle|square|rounded)$")
    headshot_placement: Optional[str] = Field(default=None, pattern=r"^(left|right)$")
    logo_placement: Optional[str] = Field(default=None, pattern=r"^(left|right)$")
    rights_and_consent_confirmed: bool = False

    @field_validator("headshot_path", "logo_path")
    @classmethod
    def reject_retired_demo_assets(cls, value: Optional[str]) -> Optional[str]:
        if value and "/demo-assets/" in value.replace("\\", "/"):
            raise ValueError("Demo assets have been retired; supply a client asset or null.")
        return value

    @model_validator(mode="after")
    def require_consent_for_assets(self):
        if (self.headshot_path or self.logo_path) and not self.rights_and_consent_confirmed:
            raise ValueError("Confirm rights_and_consent_confirmed before using an uploaded asset.")
        return self
