"""This is to test the dataset_class"""
import torch
from torchvision import transforms

from digit_recognizer.dataset.kaggle_mnist import KaggleMNISTDataset


def test_kaggle_mnist_class():
    """This is to test the data loading"""
    from digit_recognizer.config import DATA_PATH

    # Train Dataset: 48k
    kg_mnist = KaggleMNISTDataset(
        DATA_PATH / "Kaggle",
        train=True,
        transform=transforms.ToTensor(),
    )
    example = kg_mnist[1]

    assert len(kg_mnist) == 42000
    # This to ensure the sample is 28x28 image
    assert example["sample"].shape == torch.Size([1, 28, 28])

    # Test Dataset: 28k data
    kg_mnist_test = KaggleMNISTDataset(
        DATA_PATH / "Kaggle",
        train=False,
        transform=transforms.ToTensor(),
    )
    assert len(kg_mnist_test) == 28000
