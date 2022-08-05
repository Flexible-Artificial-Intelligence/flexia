![Flexia logo](assets/images/flexia_logo.png)


<center>
English
|
<a href="README_ru.md">Русский</a>

</center><br>

Flexia is an open-source library that provides a high-level API for developing Machine Learning and Deep Learning models for almost all kinds of tasks in a few lines of code. 

The main purpose of Flexia is to provide a comfortable interface, flexible functionality, and readable code, thereby allowing users to concentrate more attention and time on model development and its improvements respectively.

Flexia is mostly based on the PyTorch framework, but furthermore supports many useful functionalities from other libraries such as 🤗Transformers, bitsandbytes, etc.


## Getting Started

## Examples

## Community

Flexia is always open to your issues and pull requests! 

## TO-DO

- [ ] Documentation (readthedocs)
- [ ] Python package (PyPi)
- [ ] Distributed Training 
- [ ] Tests

###  Accelerators
- [ ] MPS Accelerator
- [ ] TPU Accelerator

### Callbacks
- [ ] Accelerator Callback
- [ ] Model Checkpoint
    - [ ] Checks for uniqueness of filename format and number of candidates.
    - [ ] Checks for other files in the output directory while overwritting.
    - [ ] Add logging logger.
    - [ ] Add aditionaly saving state dicts.
    - [ ] Multi-values checker.
- [ ] Early Stopping
    - [x] Checks for infinite values during training.
    - [ ] Add logging logger.
    - [ ] Multi-values checker.


### Loggers
- [ ] DataFrame Logger
- [ ] TensorBoard Logger
- [ ] Weights & Biases Logger
    - [ ] Summary values: auto (checks the difference during training) and others.
    - [ ] Log accelerators statistics. 
    - [ ] Custom list of logging values.
- [ ] Print Logger
    - [ ] User defined formatting
    - [ ] Log accelerators statistics.
    - [ ] Custom list of logging values.
- [ ] Logging Logger 
    - [ ] User defined formatting
    - [ ] Log accelerators statistics.
    - [ ] Custom list of logging values.
- [ ] Keras Logger
    - [ ] Log accelerators statistics. 
    - [ ] Custom list of logging values.

### Trainer
- [ ] Resuming training, validation and inference history values