"""This is Dataset Interfece"""
from typing import Any, Dict

import torch


class Dataset_Interface(torch.utils.data.Dataset):
    """This is the dataset interface"""

    def __init__(self) -> NotImplementedError:
        """Abstract init

        Raises:
            NotImplementedError: _description_

        Returns:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def __len__(self) -> NotImplementedError:
        """Abstract Length function"""
        raise NotImplementedError

    def __getitem__(self, index: int) -> NotImplementedError:
        """Abstract getitem function"""
        raise NotImplementedError

    def __repr__(self) -> str(Dict[str, Any]):
        """Repr function"""
        return str(
            {
                key: value
                for key, value in zip(
                    ["Module", "Name", "ObjectID"],
                    [self.__module__, type(self).__name__, hex(id(self))],
                )
            }
        )
