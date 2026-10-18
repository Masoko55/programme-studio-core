from pydantic import BaseModel, Field, field_validator, model_validator


class LayoutZone(BaseModel):
    x: float = Field(
        ge=0.0,
        le=1.0,
    )

    y: float = Field(
        ge=0.0,
        le=1.0,
    )

    width: float = Field(
        gt=0.0,
        le=1.0,
    )

    height: float = Field(
        gt=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.x + self.width > 1.0:
            raise ValueError(
                "Layout zone exceeds page width."
            )

        if self.y + self.height > 1.0:
            raise ValueError(
                "Layout zone exceeds page height."
            )

        return self


class LayoutGuidance(BaseModel):
    title_zone: LayoutZone

    programme_zone: LayoutZone

    headshot_zone: LayoutZone | None = None

    logo_zone: LayoutZone | None = None


class CreativeDirectionOutput(BaseModel):
    composition_strategy: str = Field(min_length=8)
    focal_strategy: str = Field(min_length=8)
    depth_strategy: str = Field(min_length=8)
    border_or_perimeter_strategy: str = Field(min_length=8)
    title_safe_region_strategy: str = Field(min_length=8)
    programme_safe_region_strategy: str = Field(min_length=8)

    @field_validator(
        "composition_strategy", "focal_strategy", "depth_strategy",
        "border_or_perimeter_strategy", "title_safe_region_strategy",
        "programme_safe_region_strategy",
    )
    @classmethod
    def require_strategy_content(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 8:
            raise ValueError("A direction strategy must contain meaningful content.")
        return value

    direction_id: str = Field(
        min_length=1,
        max_length=1,
    )

    role: str = Field(
        min_length=1,
    )

    positive_prompt: str = Field(
        min_length=1,
    )

    negative_prompt: str = Field(
        min_length=1,
    )

    title_suggestions: list[str]

    short_design_rationale: str = Field(
        min_length=1,
    )

    layout_guidance: LayoutGuidance
