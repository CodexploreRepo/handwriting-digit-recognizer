# Handwriting Digit Recognizer using Deep Learning

- Inspiration: [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/)



## Result Evaluation

|  **Model**          |  **On the Validation set**  | **On the test set** |
|:-------------------:|:---------------------------:|:-------------------:|
|  Mobilenet          |          99.63%             |       99.68%        |
|  VGG16              |          99.61%             |       99.68%        |
|  Resnet164          |        **99.72%**           |       99.70%        |
|  WideResnet28-10    |        **99.72%**           |     **99.76%**      |

|  **Ensemble (all)** |  **On the Validation set**  | **On the test set** |
|:-------------------:|:---------------------------:|:-------------------:|
|  Unweighted average |          99.70%             |       99.75%        |
|  Majority voting    |          99.71%             |       99.76%        |
|  Super Learner      |        **99.73%**           |     **99.77%**      |

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
