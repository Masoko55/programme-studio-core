from typing import Optional

from pydantic import BaseModel, Field


class PromptJobRequest(BaseModel):


    event_type: str = Field(..., min_length=1)
    theme: str = Field(..., min_length=1)
    age_group: str = Field(..., min_length=1)
    creative_description: str = Field(..., min_length=1)
    primary_colour: Optional[str] = None
    secondary_colour: Optional[str] = None
    title_preference: Optional[str] = None
    event_date: Optional[str] = None
    start_time: Optional[str] = None
    timezone: Optional[str] = None
    output_language: str = Field(default="en", min_length=1)
    accessibility_preferences: list[str] = Field(default_factory=list)
