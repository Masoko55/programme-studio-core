from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    programme_data_path: Path = Path("/data/programmes")
    image_service_port: int = 8002
    engine_1_id: str = "flux-2"
    engine_1_label: str = "FLUX.2"
    engine_2_id: str = "sdxl-1-0"
    engine_2_label: str = "SDXL 1.0"
    engine_3_id: str = "sd-3-5-medium"
    engine_3_label: str = "Stable Diffusion 3.5 Medium"
    comfyui_base_url: str = "http://192.168.68.115:8188"
    comfyui_connect_timeout_seconds: float = Field(default=10, gt=0)
    comfyui_generation_timeout_seconds: float = Field(default=1800, gt=0)
    comfyui_poll_interval_seconds: float = Field(default=2, gt=0)
    max_candidate_retries: int = Field(default=2, ge=0, le=2)
    comfyui_workflow_path: Path = SERVICE_ROOT / "workflows"
    flux2_model: str = "flux-2-klein-4b.safetensors"
    flux2_clip: str = "qwen_3_4b.safetensors"
    flux2_vae: str = "flux2-vae.safetensors"
    sdxl_checkpoint: str = "sd_xl_base_1.0.safetensors"
    sd35_checkpoint: str = "sd3.5_medium.safetensors"
    sd35_clip_l: str = "clip_l.safetensors"
    sd35_clip_g: str = "clip_g.safetensors"
    sd35_t5: str = "t5xxl_fp8_e4m3fn.safetensors"
    ollama_base_url: str = "http://192.168.68.115:11434"
    gpu_lease_timeout_seconds: float = 1800
    repository_service_base_url: str = "http://localhost:8003"
    repository_connect_timeout_seconds: float = 10
    repository_read_timeout_seconds: float = 300
    generation_width: int = Field(default=1024, ge=256, le=2048, multiple_of=16)
    generation_height: int = Field(default=1408, ge=256, le=2048, multiple_of=16)
    final_image_width: int = 2480
    final_image_height: int = 3508
    final_image_dpi: int = 300
    composer_margin_x: int = 119
    composer_margin_y: int = 119

    @property
    def gpu_lease_path(self) -> Path:
        return self.programme_data_path / ".gpu.lock"

    @model_validator(mode="after")
    def validate_configuration(self):
        if not self.programme_data_path.is_absolute():
            raise ValueError("PROGRAMME_DATA_PATH must be absolute")
        ids = [self.engine_1_id, self.engine_2_id, self.engine_3_id]
        import re
        if len(set(ids)) != 3 or not all(re.fullmatch(r"[a-z0-9][a-z0-9-]*", x) for x in ids):
            raise ValueError("Engine IDs must be three distinct safe slugs")
        if (self.final_image_width, self.final_image_height, self.final_image_dpi) != (2480, 3508, 300):
            raise ValueError("Final output must be 2480x3508 at 300 DPI")
        return self

    model_config = SettingsConfigDict(env_file=SERVICE_ROOT / ".env", env_file_encoding="utf-8")

settings = Settings()
