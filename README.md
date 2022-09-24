# Handwriting Digit Recognizer using Deep Learning

- Inspiration: [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/)

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

|    **Model**    | **On the Validation set** | **On the test set** |
| :-------------: | :-----------------------: | :-----------------: |
|    Mobilenet    |          99.63%           |       99.68%        |
|      VGG16      |          99.61%           |       99.68%        |
|    Resnet164    |        **99.72%**         |       99.70%        |
| WideResnet28-10 |        **99.72%**         |     **99.76%**      |

| **Ensemble (all)** | **On the Validation set** | **On the test set** |
| :----------------: | :-----------------------: | :-----------------: |
| Unweighted average |          99.70%           |       99.75%        |
|  Majority voting   |         99.71%            |       99.76%        |
|   Super Learner    |        **99.73%**         |     **99.77%**      |

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
