"""Models Created
1. BasicConvNet (3 basic Convolution layers, 2 MaxPooling layers, 1 AvgPooling layer,
                3 fc layers)
2.
"""
import torch
from torch import nn
from torch.nn import functional as F

from digit_recognizer.models.interface import Model_Interface


class BasicConvNet(Model_Interface):
    """BasicConvNet
    1. 3 Conv2d layers with BatchNorm
    2. 2 MaxPool layers
    3. 1 AvgPool layers connecting to final Conv2d layer
    4. 3 fc layers with Dropout
    """

    def __init__(self, num_classes=10, lr=1e-4, dropout_rate=0.3):
        super(BasicConvNet, self).__init__(num_classes, lr, dropout_rate)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=128, out_features=64, bias=False)
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=False)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes, bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=128)

    def forward(self, inputs):
        """Model Forward Propagation step

        Args:
            inputs (torch.Tensor). Dimension = (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor. Dimension = (batch_size, 10)
        """
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        out = self.fc3(x)
        return out

    def configure_optimizers(self):
        """Pytorch Lightning setup for optimizer

        Returns:
            torch optimizer: inherits hparams["lr"] for learning rate hyperparameters
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        """Pytorch Lightning training step. This step involves reseting gradient,
        updating loss/gradient for a single batch and perform backward propagation
        to update the model weights. Done on train_dataloaders.

        Args:
            batch: Default Pytorch Lightning structure. No changes needed.
            batch_idx: Default Pytorch Lightning structure. No changes needed.

        Returns:
            loss: loss for a single training batch
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("training_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch validatation step on validation DataLoader

        Args:
            batch: Default Pytorch Lightning structure. No changes needed.
            batch_idx: Default Pytorch Lightning structure. No changes needed.

        Returns:
            loss: loss for a single validation batch
        """
        x, y = batch
        logits = self(x)
        val_loss = self.loss_fn(logits, y)

        prediction = torch.argmax(logits, 1)
        acc = (prediction == y).sum() / len(y)
        self.log_dict(
            {"val_acc": acc, "val_loss": val_loss}, prog_bar=True, logger=True
        )
        return val_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """Predict Step for inference
        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
            dataloader_idx (int, optional): _description_. Defaults to 0.

        Returns:
            torch.Tensor: Size = (batch_size, num_classes)
        """
        logits = self(batch)
        logits = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return preds
