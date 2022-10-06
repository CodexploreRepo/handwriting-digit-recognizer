"""Digit Recognition Router definition
"""
import pathlib

from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image

from backend.app.config import Settings, get_settings
from digit_recognizer.inference.predict import Predictor

model_path = (
    (pathlib.Path(__file__).resolve().parents[0])
    / "deploy"
    / "pretrained_resnet50.ckpt"
)

predictor = Predictor(str(model_path))

router = APIRouter(prefix="/digit_recognizer")


@router.get("/")
def hello():
    """base endpoint"""
    return {"Model Name": "pretrained_resnet50"}


@router.get("/env")
def get_env(settings: Settings = Depends(get_settings)):
    """check env"""
    return {"env": settings.environment}


@router.post("/predict")
def upload(file: UploadFile = File(...)):
    """Get Prediction

    Args:
        file (UploadFile, optional): Upload Image file. To convert into PIL Image and use
        loaded model to serve prediction

    Returns:
        Dict: {Predicted Class, Predicted Class Probability}
    """
    try:
        sample = Image.open(file.file)
        output = predictor.predict(sample)
        return output

    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
