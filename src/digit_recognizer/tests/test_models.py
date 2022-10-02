"""This module is to test Model
"""
import torch

from digit_recognizer.models.conv_net import BasicConvNet
from digit_recognizer.models.pretrain import *


def test_model():
    """Test that model outputs correct size"""
    test_dict = {(1, 28, 28): [BasicConvNet()], (3, 28, 28): [PretrainedResNet50()]}

    for img_size in test_dict.keys():
        models = test_dict[img_size]
        c, h, w = img_size
        sample = torch.randn(3, c, h, w)
        for model in models:
            out_sample = model(sample)
            assert out_sample.shape == torch.Size([3, 10]), "Output Size mismatch"
