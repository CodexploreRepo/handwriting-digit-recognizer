from app.config import Settings, get_settings
from app.schemas import AddPayload
from fastapi import APIRouter, Depends

from digit_recognizer import config

router = APIRouter(prefix="/hello")


@router.get("/")
def hello():
    return {"Hello": config.PATH}


@router.post("/add")
def add(body: AddPayload):
    return {"new_number": body.first_number + body.second_number}


@router.get("/env")
def get_env(settings: Settings = Depends(get_settings)):
    return {"env": settings.environment}
