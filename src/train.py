"""This module is to train the models
"""
import argparse
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, model_checkpoint
from pytorch_lightning.profiler import SimpleProfiler

from digit_recognizer.config import MODEL_PARAMS, MODEL_PATH
from digit_recognizer.datamodule.mnist import KaggleMNISTDataModule
from digit_recognizer.utils import seed_everything


def get_argument_parser():
    """
    Argument parser which returns the options which the user inputted.
    Arguments:
    - Model
    - No of epochs
    - Learning rate
    - Batch size

    Returns:
        argparse.ArgumentParser().parse_args()
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        help="Choice of model (default: basic_conv_net)",
        type=str,
        default="basic_conv_net",
    )

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

    parser.add_argument(
        "-v",
        "--version",
        help="ckpt location to load from (default: -1 (No checkpoint load))",
        type=int,
        default=-1,
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
    model_name = args.model
    ver = args.version

    seed_everything()

    print(
        f"No of epochs: {epochs} \n Batch size: {batch_size} \n Learning rate: {learning_rate}"
    )
    data_module = KaggleMNISTDataModule(
        batch_size=batch_size,
        rbg=MODEL_PARAMS[model_name]["rbg"],
    )

    if model_name in MODEL_PARAMS:
        model = MODEL_PARAMS[model_name]["model"](lr=learning_rate)
    else:
        raise Exception("Model Not Setup. Please configure your model in config.py")

    version = "version_" + str(ver)
    cpkt_path = MODEL_PATH / "lightning_logs" / version
    if cpkt_path.exists():
        model_path = [
            f
            for f in os.listdir(str(cpkt_path) + "/checkpoints")
            if f.endswith(".ckpt")
        ][0]
        if model_path.split("-")[0] == model_name:
            model = model.load_from_checkpoint(
                str(cpkt_path) + "/checkpoints/" + model_path
            )
            print(f"Checkpoint successfully loaded on {model_name} using {version}")
        else:
            print("Model type mismatched. No checkpoint loaded")
    else:
        print("No ckpt_path found. No checkpoint loaded")

    profiler = SimpleProfiler()

    model_checkpoint_callback = model_checkpoint.ModelCheckpoint(
        filename=model_name + "-{epoch}-{step}-loss_{val_loss:.2f}-acc_{val_acc:.5f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",  # stop when get a min val_loss
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
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
    ## python3 train.py -m basic_conv_net -v 0 -e 3
    main()
