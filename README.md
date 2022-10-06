# Handwriting Digit Recognizer using Deep Learning

- Inspiration: [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/)
- FastAPI Image (ResNet50): https://hub.docker.com/repository/docker/lequan2902/mnist_digit:latest

## Installation

From the source directory run the following commands

### Virtual Env Creation & Activation

- `python3 -m venv venv` for initialising the virtual environment
- `source venv/bin/activate` for activating the virtual environment

### Dependency Installation

The following commands shall be ran **after activating the virtual environment**.

- `pip install --upgrade pip` for upgrading the pip
- `pip install -r requirements.txt` for the functional dependencies
- `pip install -r requirements-dev.txt` for the development dependencies. (should include `pre-commit` module)
- `pre-commit install` for installing the precommit hook

For the extra modules, which are not a standard pip modules (either from your own `src` or from any github repo)

- `pip install -e .` for the files/modules in `src` to be accessed as a package. This is accompanied with `setup.py` and `setup.cfg` files
  - `-e` means installing a project in _editable_ mode, thus any local modifications made to the code will take effect without reinstallation.

## Result Evaluation

|    **Model**    | **On the Validation set** | **On Kaggle set** |
| :-------------: | :-----------------------: | :---------------: |
|   Basic Conv    |         **100%**          |      **99%**      |
|    Mobilenet    |        **xx.xx%**         |    **xx.xx%**     |
|      VGG16      |        **xx.xx%**         |    **xx.xx%**     |
|    Resnet50     |        **99.00%**         |    **98.85%**     |
|    Resnet164    |        **xx.xx%**         |    **xx.xx%**     |
| WideResnet28-10 |        **xx.xx%**         |    **xx.xx%**     |

## Pytorch Lightning

- To activate Tensorboard: `tensorboard --logdir=model_chkpt/lightning_logs/`

## Training Methodology

### Model Architectures

- [Mobilenet](https://arxiv.org/abs/1704.04861)
- [VGG16](https://arxiv.org/abs/1409.1556)
- [Resnet164](https://arxiv.org/abs/1603.05027)
- [WideResnet28-10](https://arxiv.org/abs/1603.05027)

### Ensembling methods

- Unweighted average
- Majority voting
- [Super Learner](https://arxiv.org/abs/1704.01664)

## Pytest

```Shell
pytest --durations=0 #Show all times for tests and setup and teardown

pytest --durations=1 #Just show me the slowest
```
