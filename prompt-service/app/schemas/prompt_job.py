from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class ProgrammeItem(BaseModel):
    time: str
    title: str
    description: Optional[str] = None


class PromptJobRequest(BaseModel):
    event_type: str = Field(..., min_length=1)
    theme: str = Field(..., min_length=1)
    age_group: str = Field(..., min_length=1)

    event_date: str
    start_time: str
    timezone: str

    primary_colour: Optional[str] = None
    secondary_colour: Optional[str] = None

    creative_description: str = Field(..., min_length=1)

    title: str = Field(..., min_length=1)
    venue: Optional[str] = None

    programme: List[ProgrammeItem]

    headshot_path: Optional[str] = None
    logo_path: Optional[str] = None

    @field_validator("headshot_path", "logo_path")
    @classmethod
    def reject_retired_demo_assets(cls, value: Optional[str]) -> Optional[str]:
        """Require a real supplied asset for production programme requests."""
        if value and "/demo-assets/" in value.replace("\\", "/"):
            raise ValueError(
                "Demo assets have been retired. Supply the approved client "
                "headshot or logo from a non-demo path."
            )
        return value
