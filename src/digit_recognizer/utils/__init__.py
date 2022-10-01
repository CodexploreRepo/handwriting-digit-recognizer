import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from digit_recognizer.config import DATA_PATH, MODEL_PATH, RESULT_PATH
from digit_recognizer.datamodule.mnist import KaggleMNISTDataModule
from digit_recognizer.models.conv_net import BasicConvNet


def seed_everything(seed: int = 2022):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def test_checkpoint(model_ckpt: str, on_train_set: bool = True):

    seed_everything()

    kaggle_data_module = KaggleMNISTDataModule(batch_size=32)
    if on_train_set == True:
        test_dataloader = kaggle_data_module.train_dataloader()
    else:
        test_dataloader = kaggle_data_module.test_dataloader()

    model = BasicConvNet()
    model_path = MODEL_PATH / model_ckpt

    model = model.load_from_checkpoint(str(model_path))

    test_pred = torch.LongTensor()
    test_labels = torch.LongTensor()

    for batch in test_dataloader:
        imgs, labels = batch

        logits = model(imgs)
        logits = F.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        test_pred = torch.concat((test_pred, pred), dim=0)
        test_labels = torch.concat((test_labels, labels), dim=0)

    print((test_pred == test_labels).sum() / len(test_pred))
