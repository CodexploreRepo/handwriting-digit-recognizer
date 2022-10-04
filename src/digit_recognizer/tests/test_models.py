"""This module is to test Models
"""
import torch

from digit_recognizer.config import MODEL_PARAMS


def test_model():
    """Test all models for correct output size.
    Register the model settings in config.py"""

    for model_name in MODEL_PARAMS:
        model = MODEL_PARAMS[model_name]["model"]()
        if MODEL_PARAMS[model_name]["rbg"]:
            sample = torch.randn(3, 3, 28, 28)
        else:
            sample = torch.randn(3, 1, 28, 28)

        out_sample = model(sample)
        assert out_sample.shape == torch.Size(
            [3, 10]
        ), f"Output Size mismatch for {model_name}"
