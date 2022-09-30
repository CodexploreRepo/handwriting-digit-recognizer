"""This is for Model Inference to Kaggle submission"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from digit_recognizer.config import DATA_PATH, MODEL_PATH, RESULT_PATH
from digit_recognizer.dataset.kaggle_mnist import KaggleMNISTDataset
from digit_recognizer.models.conv_net import BasicConvNet
from digit_recognizer.utils import seed_everything

seed_everything()


def inference():
    """Model Inference"""
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5),
                (0.5),
            ),
        ]
    )
    kaggle_mnist_ds = KaggleMNISTDataset(
        data_path=DATA_PATH / "Kaggle", train=False, transform=transform
    )
    test_dl = DataLoader(kaggle_mnist_ds, batch_size=32, shuffle=False)

    model = BasicConvNet()
    model_path = (
        MODEL_PATH / "epoch=7-step=10392-loss_val_loss=0.01-acc_val_acc=1.00.ckpt"
    )
    model.load_from_checkpoint(str(model_path))

    test_pred = torch.LongTensor()

    for batch in tqdm(test_dl):
        imgs, _ = batch

        logits = model(imgs)
        # logits = F.log_softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        test_pred = torch.concat((test_pred, pred), dim=0)

    out_df = pd.DataFrame(
        np.c_[np.arange(1, 28000 + 1)[:, None], test_pred.numpy()],
        columns=["ImageId", "Label"],
    )
    out_df.head()
    out_df.to_csv(RESULT_PATH / "submission.csv", index=False)


if __name__ == "__main__":
    inference()
