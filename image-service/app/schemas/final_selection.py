from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FinalSelectionRequest(BaseModel):
    """The client's selected candidate and optional supplied finishing assets."""

    engine_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    direction_id: str = Field(pattern=r"^[ABC]$")
    headshot_path: Optional[str] = None
    logo_path: Optional[str] = None

    @field_validator("headshot_path", "logo_path")
    @classmethod
    def reject_retired_demo_assets(cls, value: Optional[str]) -> Optional[str]:
        if value and "/demo-assets/" in value.replace("\\", "/"):
            raise ValueError("Demo assets have been retired; supply a client asset or null.")
        return value
