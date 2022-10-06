"""This file is for Digit Recognizer Router"""
import io

from fastapi import APIRouter, File, UploadFile
from PIL import Image

router = APIRouter(prefix="/digit_recognizer")


@router.post("/predict")
def upload(file: UploadFile = File(...)):
    """Upload

    Args:
        file (UploadFile, optional): _description_. Defaults to File(...).

    Returns:
        _type_: _description_
    """
    try:
        contents = file.file.read()
        # io.BytesIO object created from the bytes object
        img = Image.open(io.BytesIO(contents))
        print(img)

    except Exception as e:
        return {"message": f"Error: {e}"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}
