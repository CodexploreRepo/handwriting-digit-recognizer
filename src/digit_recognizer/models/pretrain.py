"""Pretrained Models for finetuning
"""
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from digit_recognizer.models.conv_net import BasicConvNet
from digit_recognizer.models.interface import Model_Interface


class PretrainedAbstract(Model_Interface):
    """Generic Pretrained Model
    Setup Replace common replace last layer method for all future Pretrained Model
    to replace final layer with fully connected layer on correct num_classes.
    """

    def __init__(
        self,
        num_classes: int = 10,
        lr: float = 1e-4,
        dropout_rate: float = 0.3,
    ):
        """Pretrained Generic Model

        Args:
            num_classes (int, optional): Number of classes. Defaults to 10.
            lr (float, optional): Learning Rate. Defaults to 1e-4.
            dropout_rate (float, optional): Dropout Rate. Defaults to 0.3.
        """
        super(PretrainedAbstract, self).__init__(num_classes, lr, dropout_rate)
        self.model = BasicConvNet()

    def forward(self, inputs):
        """Abstract forward function"""
        return

    def replace_last_layer(self):
        """Class Method to replace last fully connected layer from the pretrained
        model
        """
        last_layer_name, last_layer = list(self.model.named_modules())[-1]
        self.featveclen = last_layer.weight.shape[1]

        if len(last_layer_name.split(".")) == 2:
            # For Inception model, last_layer_name = inception.fc
            # Replace last layer with nn.Identity(), model.inception.fc = nn.Idenity()
            exec(
                "self.model.%s[%s] = nn.Identity()"
                % (last_layer_name.split(".")[0], last_layer_name.split(".")[1])
            )
        else:
            # For other pretrained model,
            # model.fc = nn.Idenity()
            exec("self.model.%s = nn.Identity()" % (last_layer_name,))

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


class PretrainedResNet50(PretrainedAbstract):
    """Pretrained ResNet50 model"""

    def __init__(
        self,
        num_classes: int = 10,
        lr: float = 1e-4,
        dropout_rate: float = 0.3,
        pretrained_weights=False,
        freeze: Union[bool, int] = False,
    ):
        """Pretrained ResNet50

        Args:
            num_classes (int, optional): Number of Classes. Defaults to 10.

            lr (float, optional): Learning Rate. Defaults to 1e-4.

            dropout_rate (float, optional): Dropout Rate. Defaults to 0.3.

            pretrained_weights (bool, optional): Whether to load pretrained weights. Defaults to False.

            freeze (Union[bool, int], optional): bool or int. If True, freeze
            weights for all layers under ResNet model. If False, do not freeze
            weights for any layers. If given int value, freeze weights of the
            first (bottom) number of layers. Defaults to False.
        """
        super(PretrainedResNet50, self).__init__(num_classes, lr, dropout_rate)
        if pretrained_weights:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = resnet50()

        # To update self.featveclen attribute
        self.replace_last_layer()

        if freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        self.fc1 = nn.Linear(self.featveclen, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, inputs):
        """Forward Propagation step

        Args:
            inputs: torch.Tensor. Dimension = (batch_size, n_channels, 28, 28)

        Returns:
            torch.Tensor. Dimension = (batch_size, num_classes)
        """
        x = self.model(inputs)
        x = F.selu(self.dropout(self.fc1(x)))
        x = F.selu(self.dropout(self.fc2(x)))
        out = self.fc3(x)
        return out
