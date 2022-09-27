"""This module is to test Model
"""
import torch

from digit_recognizer.models.models import BasicConvNet


def test_model():
    """Test that model outputs correct size"""
    sample = torch.randn(3, 1, 28, 28)
    model = BasicConvNet()
    out_sample = model(sample)

    assert out_sample.shape == torch.Size([3, 10]), "Output Size mismatch"
