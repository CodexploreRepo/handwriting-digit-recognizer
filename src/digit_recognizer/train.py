"""This module is to train the models
"""
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, model_checkpoint
from pytorch_lightning.profiler import SimpleProfiler

from digit_recognizer.config import MODEL_PATH
from digit_recognizer.datamodule.mnist import KaggleMNISTDataModule
from digit_recognizer.models.conv_net import BasicConvNet
from digit_recognizer.utils import seed_everything

seed_everything()


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
    data_module.setup()

    model = BasicConvNet(lr=learning_rate)
    model_path = MODEL_PATH / "acc_97.ckpt"
    if model_path.exists():
        model = model.load_from_checkpoint(model_path)

    profiler = SimpleProfiler()

    model_checkpoint_callback = model_checkpoint.ModelCheckpoint(
        filename="{epoch}-{step}-loss_{val_loss:.2f}-acc_{val_acc:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",  # stop when get a min val_loss
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min",  # stop when get a min val_loss
    )

    trainer = Trainer(
        max_epochs=epochs,
        default_root_dir=MODEL_PATH,  # where the lightning_log is stored
        # accelerator="gpu",
        log_every_n_steps=100,
        callbacks=[early_stop_callback, model_checkpoint_callback],
        profiler=profiler,
        # precision=16,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
