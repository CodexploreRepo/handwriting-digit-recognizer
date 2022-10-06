"""Generate Test Client for testing
"""
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseSettings

from backend.app.application_factory import create_app
from backend.app.config import Settings, get_settings


def get_settings_override() -> BaseSettings:
    """_summary_

    Returns:
        BaseSettings: _description_
    """
    return Settings(environment="testing")


@pytest.fixture(scope="module")
def test_app():
    """Generate FastAPI Application

    Yields:
        _type_: _description_
    """
    app = create_app()
    app.dependency_overrides[get_settings] = get_settings_override

    with TestClient(app) as test_client:
        yield test_client
