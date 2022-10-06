"""create FastAPI and include router
"""
from fastapi import FastAPI

from backend.app.routers import get_prediction


def create_app() -> FastAPI:
    app = FastAPI()
    #  app.include_router(healthcheck.router)
    app.include_router(get_prediction.router)

    @app.get("/")
    def home_page():
        return {"Service": "Computer Vision tasks"}

    return app


if __name__ == "__main__":
    create_app()
