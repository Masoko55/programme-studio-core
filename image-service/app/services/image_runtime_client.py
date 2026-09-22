"""Compatibility imports for callers of the retired custom runtime client."""
from app.services.comfyui_client import generate_remote_image, get_runtime_health

__all__ = ["generate_remote_image", "get_runtime_health"]
