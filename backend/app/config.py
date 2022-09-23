from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    environment: str = "dev"


@lru_cache
def get_settings() -> BaseSettings:
    return Settings()
