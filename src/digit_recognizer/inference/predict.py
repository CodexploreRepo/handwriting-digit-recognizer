"""Module for API Inference
"""
import torch
import torch.nn.functional as F
from torchvision import transforms

from digit_recognizer.config import MODEL_PARAMS


class Predictor:
    """Predictor Class"""

    def __init__(self, ckpt_path: str):
        """_summary_

        Args:
            ckpt_path (str): Checkpoint path to load model.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.ckpt_path = ckpt_path
        self.model_type = ckpt_path.split("/")[-1].split(".")[0]
        model = MODEL_PARAMS[self.model_type]["model"]
        self.model = model.load_from_checkpoint(self.ckpt_path)

        if MODEL_PARAMS[self.model_type]["rbg"]:
            self.img_setting = "RGB"
        else:
            self.img_setting = "L"

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(28),
                transforms.Normalize(
                    (0.5),
                    (0.5),
                ),
            ]
        )

    def predict(self, input):
        """_summary_

        Args:
            input (PIL Image): Input PIL Img

        Returns:
            Dict: Prediction & Softmax probability of prediction
        """
        input = self.transform(input.convert(self.img_setting)).unsqueeze(0)
        input = input.to(self.device)
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            out = self.model(input)
        logits = F.softmax(out, dim=1)
        probs = torch.max(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        return {"prediction": pred.item(), "probs": probs[0].item()}
