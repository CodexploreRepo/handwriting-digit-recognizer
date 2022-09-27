"""This is for plot uitl"""
import random

import matplotlib.pyplot as plt
import torch


def plot_sample(ds: torch.utils.data.Dataset) -> None:
    """This is to plot the sample

    Args:
        ds (torch.utils.data.Dataset): Pytorch Dataset Class
    """
    n_cols, n_rows = 4, 2

    idx_list = list(range(len(ds)))
    _, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    for ax_r in axes:
        for ax in ax_r:
            sample = ds[idx_list.pop(random.randint(0, len(idx_list) - 1))]
            ax.imshow(sample["sample"], cmap="gray")

            ax.set_title(sample["target"])

    plt.show()
