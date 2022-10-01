"""This is for Model Inference to Kaggle submission"""
import argparse

import numpy as np
import pandas as pd
import torch

# import torch.nn.functional as F
from pytorch_lightning import Trainer

from digit_recognizer.config import MODEL_PATH, RESULT_PATH
from digit_recognizer.datamodule.mnist import KaggleMNISTDataModule
from digit_recognizer.models.conv_net import BasicConvNet
from digit_recognizer.utils import seed_everything

# from tqdm import tqdm


def get_argument_parser():
    """
    Argument parser which returns the options which the user inputted.
    Arguments:
    - Version of the checkpoint

    Returns:
        argparse.ArgumentParser().parse_args()
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v",
        "--version",
        help="Version of the checkpoint (default: 0)",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    return args


def inference_kaggle(ver: int):
    """Model Inference"""

    seed_everything()

    kaggle_data_module = KaggleMNISTDataModule(batch_size=32)

    # test_dataloader = kaggle_data_module.predict_dataloader()

    version = "version_" + str(ver)
    ckpt_folder = MODEL_PATH / "lightning_logs" / version

    if not ckpt_folder.exists():
        raise Exception("Invalid Checkpoint")
    else:
        file_list = [str(file) for file in (ckpt_folder / "checkpoints").glob("*")]
        model_path = [file for file in file_list if file.endswith(".ckpt")][0]

    model = BasicConvNet()
    # model = model.load_from_checkpoint(model_path)

    trainer = Trainer(deterministic=True)
    preds = trainer.predict(model, kaggle_data_module, ckpt_path=model_path)
    test_pred = torch.concat(preds, dim=0)

    # test_pred = torch.LongTensor()

    # for imgs in tqdm(test_dataloader):
    #     logits = model(imgs)
    #     logits = F.softmax(logits, dim=1)
    #     pred = torch.argmax(logits, dim=1)
    #     test_pred = torch.concat((test_pred, pred), dim=0)

    out_df = pd.DataFrame(
        np.c_[np.arange(1, 28000 + 1)[:, None], test_pred.numpy()],
        columns=["ImageId", "Label"],
    )
    out_df.head()
    out_df.to_csv(RESULT_PATH / "submission.csv", index=False)


if __name__ == "__main__":
    args = get_argument_parser()
    version = args.version
    inference_kaggle(version)
