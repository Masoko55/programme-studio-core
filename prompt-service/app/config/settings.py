from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    ollama_base_url: str = "http://192.168.68.115:11434"

    programme_data_path: Path = Path(
        "/data/programmes"
    )

    ollama_connect_timeout_seconds: float = 10
    ollama_read_timeout_seconds: float = 900
    direction_a_model: str = "qwen2.5:14b"
    direction_a_role: str = "elegant"
    direction_b_model: str = "gemma3:27b"
    direction_b_role: str = "expressive"
    direction_c_model: str = "mistral-small3.1:24b"
    direction_c_role: str = "contemporary"
    comfyui_base_url: str = "http://192.168.68.115:8188"
    image_service_base_url: str = "http://programme-image:8002"
    image_service_connect_timeout_seconds: float = 10
    image_service_read_timeout_seconds: float = 3600
    gpu_lease_timeout_seconds: float = 1800

    @property
    def gpu_lease_path(self) -> Path:
        return self.programme_data_path / ".gpu.lock"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
