"""This module contains Interface for DataModules
"""

import pathlib
from typing import Union

import pytorch_lightning as pl

from digit_recognizer.config import DATA_PATH, NUM_WORKERS


class DataModule_Interface(pl.LightningDataModule):
    """
    LightningDataModule Interface
    """

    def __init__(
        self,
        data_dir: Union[pathlib.Path, str] = DATA_PATH,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = NUM_WORKERS,
    ) -> None:
        """
        Args:
            data_dir (str, optional): Download Data Directory. Defaults to "./data".
            batch_size (int, optional): Batch Size used to DataLoader. Defaults to 32.
            val_split (float, optional): Percentage Split between train_dataset and val_dataset. Defaults to 0.2.
        """
        self.data_dir = (
            data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        )
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        # TODO: add comment for this attrbute
        self.save_hyperparameters()
        self.prepare_data_per_node = False

    def prepare_data(self) -> None:
        """
        Download/Load data to self.data_dir
        """
        pass

    def setup(self, stage: str = "train") -> NotImplementedError:
        """_summary_

        Args:
            stage (str): "fit" or "test". Stage "fit" will split full train dataset into train set and validation set
            and store both datasets as class instances attributes. Stage "test" will return full test dataset.
        """
        raise NotImplementedError

    def train_dataloader(self) -> NotImplementedError:
        """
        Return Train DataLoader using specified batch_size
        """
        raise NotImplementedError

    def val_dataloader(self) -> NotImplementedError:
        """
        Return Validation DataLoader using specified batch_size
        """
        return NotImplementedError

    def test_dataloader(self) -> NotImplementedError:
        """
        Return Test DataLoader using specified batch_size
        """
        return NotImplementedError
