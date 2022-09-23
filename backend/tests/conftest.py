import pytest
from app.application_factory import create_app
from app.config import Settings, get_settings
from fastapi.testclient import TestClient
from pydantic import BaseSettings


def get_settings_override() -> BaseSettings:
    return Settings(environment="testing")


@pytest.fixture(scope="module")
def test_app():
    app = create_app()
    app.dependency_overrides[get_settings] = get_settings_override

    with TestClient(app) as test_client:
        yield test_client
