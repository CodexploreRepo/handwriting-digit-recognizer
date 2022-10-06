"""This file is main application"""
from app.routers import digit_recognizer
from fastapi import FastAPI


def create_app() -> FastAPI:
    """Main Application

    Returns:
        FastAPI: _description_
    """
    app = FastAPI()
    app.include_router(digit_recognizer.router)
    return app
