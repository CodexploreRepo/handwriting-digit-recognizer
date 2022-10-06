"""Unittest for inference class
"""
import pathlib

from PIL import Image

from digit_recognizer.inference.predict import Predictor


def test_predictor():
    """Input 1 image predict and generate prediction and probability"""
    model_path = (
        pathlib.Path(__file__).resolve().parents[0]
    ) / "pretrained_resnet50.ckpt"
    img_path = (pathlib.Path(__file__).resolve().parents[0]) / "23508-7.png"
    predictor = Predictor(str(model_path))
    sample = Image.open(str(img_path))

    output = predictor.predict(sample)
    print(output)
    assert output["prediction"] in list(range(10)), "Invalid Prediction"
    assert (output["probs"] <= 1) and (output["probs"] >= 0), "Invalid Probability"
