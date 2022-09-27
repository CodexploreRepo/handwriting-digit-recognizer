"""This is for Kaggle MNIST Dataset"""
import pathlib
from typing import Callable, Dict, Optional

import pandas as pd
import torch
from PIL import Image

from digit_recognizer.dataset.interface import Dataset_Interface


class KaggleMNISTDataset(Dataset_Interface):
    """Kaggle MNIST Dataset

    Args:
        Dataset_Interface (_type_): inherit from Dataset Interface
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Init method

        Args:
            data_path (pathlib.Path): the path to Kaggle Data
            train (bool, optional): _description_. Defaults to True.
            transform (Optional[Callable], optional): _description_. Defaults to None.
            target_transform (Optional[Callable], optional): _description_. Defaults to None.
        """
        self.train = train
        self.data_path = data_path / f'{"train" if self.train else "test"}.csv'
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv(self.data_path)

        if self.train:
            self.data = self.df.loc[:, self.df.columns != "label"]
            self.targets = self.df["label"]
        else:
            self.data = self.df

    def __len__(self) -> int:
        """Length

        Returns:
            int: Dataset Length
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.tensor]:
        """Get Item

        Args:
            index (int): index of the item in the dataset

        Returns:
            Dict[str, torch.tensor]: _description_
        """
        # @: list of all transform func on single sample of data
        # @: index : { sample, specific_target }

        sample = self.data.iloc[index, :].values.reshape(28, 28)
        sample = Image.fromarray(sample, mode="L")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.train:
            target = int(self.targets[index])
            if self.target_transform is not None:
                target = self.target_transform(target)

        return (
            {
                "sample": sample,
                "target": target,
            }
            if self.train
            else {"sample": sample}
        )


if __name__ == "__main__":
    from digit_recognizer.config import DATA_PATH
    from digit_recognizer.utils.plot_dataset_sample import plot_sample

    kg_mnist = KaggleMNISTDataset(DATA_PATH / "Kaggle", train=True)
    plot_sample(kg_mnist)
