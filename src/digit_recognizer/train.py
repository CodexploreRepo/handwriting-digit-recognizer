"""This module is to train the models
"""
import argparse

from pytorch_lightning import Trainer

from digit_recognizer.config import MODEL_PATH
from digit_recognizer.datamodule.mnist import KaggleMNISTDataModule
from digit_recognizer.models.conv_net import BasicConvNet


def get_argument_parser():
    """
    Argument parser which returns the options which the user inputted.
    Arguments:
    - No of epochs
    - Learning rate
    - Batch size

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
        default=32,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="The learning rate used for optimizer (default: 1e-4)",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()
    return args


def main():
    """This is for training models"""
    # load all arguments
    args = get_argument_parser()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    print(
        f"No of epochs: {epochs} \n Batch size: {batch_size} \n Learning rate: {learning_rate}"
    )
    data_module = KaggleMNISTDataModule(batch_size=batch_size)
    data_module.setup(stage="fit")

    model = BasicConvNet(lr=learning_rate).load_from_checkpoint(
        MODEL_PATH / "lightning_logs/version_0/checkpoints/epoch=2-step=2346.ckpt"
    )
    trainer = Trainer(max_epochs=epochs, default_root_dir=MODEL_PATH)

    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
