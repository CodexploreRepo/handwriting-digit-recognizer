"""This is for plot uitl"""
import random
from typing import List, Union

import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def plot_sample(ds: Union[Dataset, List], train: bool = True) -> None:
    """This is to plot the sample

    Args:
        ds (torch.utils.data.Dataset): Pytorch Dataset Class
    """

    n_cols, n_rows = 4, 2

    idx_list = list(range(len(ds)))
    _, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    for ax_r in axes:
        for ax in ax_r:
            img, label = ds[idx_list.pop(random.randint(0, len(idx_list) - 1))]
            ax.imshow(img, cmap="gray")
            if train:
                ax.set_title(label)

    plt.show()


def save_sample(ds: Union[Dataset, List], num_samples=10) -> None:
    """This is to save the samples

    Args:
        ds (torch.utils.data.Dataset): Pytorch Dataset Class
    """

    idx_list = list(range(len(ds)))
    idx = random.randint(0, len(idx_list) - 1)
    for _ in range(num_samples):
        img, label = ds[idx_list.pop(idx)]
        plt.imshow(img, cmap="gray")
        plt.savefig(f"{idx}-{label}.png")
