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


from fastapi import File, UploadFile


@router.post("/upload")
def upload(file: UploadFile = File(...)):
    """Upload

    Args:
        file (UploadFile, optional): _description_. Defaults to File(...).

    Returns:
        _type_: _description_
    """
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}
