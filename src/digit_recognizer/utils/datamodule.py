"""This module contains DataModules
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    """
    LightningDataModule for MNIST dataset
    """

    def __init__(self, data_dir: str = "./data", batch_size: int = 32, val_split=0.2):
        """
        Args:
            data_dir (str, optional): Download Data Directory. Defaults to "./data".
            batch_size (int, optional): Batch Size used to DataLoader. Defaults to 32.
            val_split (float, optional): Percentage Split between train_dataset and val_dataset. Defaults to 0.2.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5),
                    (0.5),
                ),
            ]
        )

        self.prepare_data()

    def prepare_data(self):
        """
        Download data to self.data_dir
        """
        MNIST(self.data_dir, download=True, train=True)
        MNIST(self.data_dir, download=True, train=False)

    def setup(self, stage: str):
        """_summary_

        Args:
            stage (str): "fit" or "test". Stage "fit" will split full train dataset into train set and validation set
            and store both datasets as class instances attributes. Stage "test" will return full test dataset.
        """
        if stage == "fit":

            self.mnist_full = MNIST(
                self.data_dir, train=True, transform=self.transforms
            )
            self.mnist_train, self.mnist_val = random_split(
                self.mnist_full, [50000, 10000]
            )

        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transforms
            )

    def train_dataloader(self):
        """
        Return Train DataLoader using specified batch_size
        """
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """
        Return Validation DataLoader using specified batch_size
        """
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Return Test DataLoader using specified batch_size
        """
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
