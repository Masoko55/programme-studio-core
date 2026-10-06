from pathlib import Path

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


SERVICE_ROOT = Path(
    __file__
).resolve().parents[2]


class Settings(
    BaseSettings
):
    programme_data_path: Path = Path(
        "/data/programmes"
    )

    image_service_port: int = 8002

    engine_1_id: str = "flux-2"
    engine_1_label: str = "FLUX.2"

    engine_2_id: str = "sdxl-1-0"
    engine_2_label: str = "SDXL 1.0"

    engine_3_id: str = "sd-3-5-medium"
    engine_3_label: str = (
        "Stable Diffusion 3.5 Medium"
    )

    comfyui_base_url: str = (
        "http://192.168.68.115:8188"
    )

    comfyui_connect_timeout_seconds: float = Field(
        default=10,
        gt=0,
    )

    comfyui_generation_timeout_seconds: float = Field(
        default=1800,
        gt=0,
    )

    comfyui_poll_interval_seconds: float = Field(
        default=2,
        gt=0,
    )

    # Initial generation + 8 retries = 9 attempts.
    max_candidate_retries: int = Field(
        default=8,
        ge=0,
        le=12,
    )

    max_candidate_retry_delay_seconds: int = Field(
        default=8,
        ge=1,
        le=30,
    )

    # Once every engine/direction has completed its first pass,
    # failed candidates receive another complete retry wave.
    #
    # 1 recovery wave:
    #   first pass    = 9 attempts
    #   recovery pass = 9 attempts
    #   max total     = 18 attempts
    failed_candidate_recovery_waves: int = Field(
        default=1,
        ge=0,
        le=3,
    )

    # -----------------------------
    # Deterministic visual QA
    # -----------------------------

    quality_sample_width: int = Field(
        default=256,
        ge=64,
        le=1024,
    )

    quality_sample_height: int = Field(
        default=352,
        ge=64,
        le=1408,
    )

    # Sobel edge threshold.
    quality_edge_threshold: float = Field(
        default=35.0,
        gt=0,
    )

    # Scanline detection.
    quality_scanline_delta_threshold: float = Field(
        default=5.0,
        gt=0,
    )

    quality_scanline_active_ratio: float = Field(
        default=0.78,
        gt=0,
        le=1,
    )

    quality_scanline_direction_ratio: float = Field(
        default=2.6,
        gt=1,
    )

    quality_scanline_min_mean_delta: float = Field(
        default=7.5,
        gt=0,
    )

    # Flat / degenerate output detection.
    quality_flat_max_std: float = Field(
        default=7.0,
        gt=0,
    )

    quality_flat_max_entropy: float = Field(
        default=2.4,
        gt=0,
    )

    quality_flat_max_edge_density: float = Field(
        default=0.015,
        ge=0,
        le=1,
    )

    # Reject a candidate if one quantised colour fills almost
    # the whole page and the image contains little structure.
    quality_max_single_colour_ratio: float = Field(
        default=0.97,
        ge=0,
        le=1,
    )

    # Central programme/title safe-region validation.
    #
    # This does not require a blank centre. It only rejects
    # a centre that is too visually dense to overlay text.
    quality_center_x_start: float = Field(
        default=0.18,
        ge=0,
        le=1,
    )

    quality_center_x_end: float = Field(
        default=0.82,
        ge=0,
        le=1,
    )

    quality_center_y_start: float = Field(
        default=0.12,
        ge=0,
        le=1,
    )

    quality_center_y_end: float = Field(
        default=0.82,
        ge=0,
        le=1,
    )

    quality_center_max_edge_density: float = Field(
        default=0.30,
        ge=0,
        le=1,
    )

    quality_center_max_local_contrast: float = Field(
        default=62.0,
        gt=0,
    )

    comfyui_workflow_path: Path = (
        SERVICE_ROOT
        / "workflows"
    )

    flux2_model: str = (
        "flux-2-klein-4b.safetensors"
    )

    flux2_clip: str = (
        "qwen_3_4b.safetensors"
    )

    flux2_vae: str = (
        "flux2-vae.safetensors"
    )

    sdxl_checkpoint: str = (
        "sd_xl_base_1.0.safetensors"
    )

    sd35_checkpoint: str = (
        "sd3.5_medium.safetensors"
    )

    sd35_clip_l: str = (
        "clip_l.safetensors"
    )

    sd35_clip_g: str = (
        "clip_g.safetensors"
    )

    sd35_t5: str = (
        "t5xxl_fp8_e4m3fn.safetensors"
    )

    ollama_base_url: str = (
        "http://192.168.68.115:11434"
    )

    gpu_lease_timeout_seconds: float = 1800

    repository_service_base_url: str = (
        "http://localhost:8003"
    )

    repository_connect_timeout_seconds: float = 10

    repository_read_timeout_seconds: float = 300

    generation_width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        multiple_of=16,
    )

    generation_height: int = Field(
        default=1408,
        ge=256,
        le=2048,
        multiple_of=16,
    )

    final_image_width: int = 2480
    final_image_height: int = 3508
    final_image_dpi: int = 300

    composer_margin_x: int = 119
    composer_margin_y: int = 119

    @property
    def gpu_lease_path(
        self,
    ) -> Path:
        return (
            self.programme_data_path
            / ".gpu.lock"
        )

    @model_validator(
        mode="after"
    )
    def validate_configuration(
        self,
    ):
        if not self.programme_data_path.is_absolute():
            raise ValueError(
                "PROGRAMME_DATA_PATH must be absolute"
            )

        ids = [
            self.engine_1_id,
            self.engine_2_id,
            self.engine_3_id,
        ]

        import re

        if (
            len(set(ids)) != 3
            or not all(
                re.fullmatch(
                    r"[a-z0-9][a-z0-9-]*",
                    item,
                )
                for item in ids
            )
        ):
            raise ValueError(
                "Engine IDs must be three distinct safe slugs"
            )

        if (
            self.final_image_width,
            self.final_image_height,
            self.final_image_dpi,
        ) != (
            2480,
            3508,
            300,
        ):
            raise ValueError(
                "Final output must be 2480x3508 at 300 DPI"
            )

        if (
            self.quality_center_x_start
            >= self.quality_center_x_end
        ):
            raise ValueError(
                "QUALITY_CENTER_X_START must be less than "
                "QUALITY_CENTER_X_END"
            )

        if (
            self.quality_center_y_start
            >= self.quality_center_y_end
        ):
            raise ValueError(
                "QUALITY_CENTER_Y_START must be less than "
                "QUALITY_CENTER_Y_END"
            )

        return self

    model_config = SettingsConfigDict(
        env_file=(
            SERVICE_ROOT
            / ".env"
        ),
        env_file_encoding="utf-8",
    )


settings = Settings()