"""main function for uvicorn to call and create FastAPI Session
"""
from backend.app.application_factory import create_app

app = create_app()
