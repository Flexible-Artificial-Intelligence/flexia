![Flexia logo](images/flexia_logo.png)

Flexia (Flexible Artificial Intelligence) is an open-source library that provides a high-level API for developing accurate Deep Learning models for all kinds of Deep Learning tasks such as Classification, Regression, Object Detection, Image Segmentation, etc. 

It offers a variety of methods for controlling (e.g Early Stopping, Model Checkpoint, Timing, etc.) and logging/monitoring (e.g Weights & Biases, Logging, Taqadum, etc.) training, validation, and inference loops.


## Installation

Flexia offers you many ways to do it depending on your setup: PyPi (pip), Google Colab, and Kaggle Kernels.

#### PyPi (pip)

```py
pip install flexia 
```

#### Google Colab

```
!git clone https://github.com/vad13irt/flexia.git
```

```py
import sys
sys.path.append("./flexia")
import flexia
```

#### Kaggle Kernels

There are many ways to install libraries in Kaggle Kernels. We will describe only one way, which is faster and more suitable for  Kernels Competitions, where submitting require disabling an Internet connection.


## Getting Started

## Examples

To speed up and improve your practical experience, Flexia provides some examples of how Flexia can be integrated to solve different kinds of Deep Learning tasks. Examples are presented as Jupyter Notebooks.

- [Digit Recognizer](examples/Digit%20Recognizer/)
- [Carvana Image Segmentation](examples/Carvana%20Image%20Masking%20Challenge/)
- [Global Wheat Detection](examples/Global%20Wheat%20Detection/)

## Contribution

Flexia is always open to your contributions (pull requests and issues)! Contribution is one of the possible ways to improve library functionality and make it easier and stronger! 

However, contributions are required to be well-documented, and the code should be readable and well-tested.


## Community

Flexia has communication channels in some popular social networks. There you can ask questions you are interested in, make friends, and get news about new releases!

- Discord
- Twitter
- Telegram

## TO-DO

- Documentation
    - [ ] Comments in the code.
    - [ ] Types Annotations
    - [ ] Docstrings 
    - [ ] Warnings and Exceptions

- Trainer
    - [ ] Resuming Trainer history (state dictionary)

- Callbacks
    - [ ] Verbosity
    - [ ] List wrapper

- Loggers
    - [ ] DataFrameLogger
    - [x] Environment variables in `WANDBLogger`
    - [ ] List wrapper

- [ ] Distributed Training
- [x] Torch XLA
- [ ] DeepSpeed integration
- [ ] PyPi package
- [ ] ONNX
- [ ] ONNXRuntime
- [ ] TensorRT
- [ ] Tests

## Releases
