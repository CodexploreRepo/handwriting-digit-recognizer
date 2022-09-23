from app.routers import hello
from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(hello.router)
    return app
