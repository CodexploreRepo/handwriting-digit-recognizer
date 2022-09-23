"""This module is to train the models
"""

import argparse


def get_argument_parser():
    """
    Argument parser which returns the options which the user inputted.

    Returns:
        argparse.ArgumentParser().parse_args()
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epochs",
        help="How many epochs you need to run (default: 10)",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="The number of images in a batch (default: 64)",
        type=int,
        default=64,
    )

    args = parser.parse_args()
    return args


def main():
    """This is for training models"""
    # load all arguments
    args = get_argument_parser()
    epochs = args.epochs
    batch_size = args.batch_size
    print(epochs, batch_size)


if __name__ == "__main__":
    main()
