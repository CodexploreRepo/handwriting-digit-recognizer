"""config file for settings
"""
from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    """BaseSettings"""

    environment: str = "dev"


@lru_cache
def get_settings() -> BaseSettings:
    """Get base settings. Cache maximum 128 previous calls."""
    return Settings()
