"""Interface for LightningModule
"""
import pytorch_lightning as pl
from torch import nn


class Model_Interface(pl.LightningModule):
    """This is the Model interface for Pytorch Lightning Models"""

    def __init__(
        self,
        num_classes: int = 10,
        lr: float = 1e-4,
        dropout_rate: float = 0.3,
    ):
        """_summary_

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 10.
            lr (float, optional): Learning Rate for Optimizer. Defaults to 1e-4.
            dropout_rate (float, optional): Dropout Rate. Defaults to 0.3.
        """
        super(Model_Interface, self).__init__()

        self.num_classes = num_classes
        self.learning_rate = lr
        self.save_hyperparameters()

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs) -> NotImplementedError:
        """Abstract forward function"""
        raise NotImplementedError

    def configure_optimizers(self) -> NotImplementedError:
        """Abstract Optimizer function"""
        return NotImplementedError

    def training_step(self, batch, batch_idx) -> NotImplementedError:
        """Abstract training_step function"""
        raise NotImplementedError
