# NumpyNet
NumpyNet is a plain numpy based framework for building simple convolutional neuronal networks. The way to use it is oriented on state of the art frameworks like Tensorflow or Keras, but is not meant to be a high performance network architecture, rather than a open piece of code to gain deep understanding of how deep neuronal networks like cnn's work.

## Requirements
### System
The System used for training was a windows i7, 32GB Ram system. Training was based on CPU since, GPU option is not included in this playground project.

### Environment
The code is written in Python 3.7 and a pipfile will install all dependencies

### Data
Training data was cifar-10-batches-py from:
http://www.cs.toronto.edu/~kriz/cifar.html

NumpyNet always assumes quadratic input data - image shape like 32x32
and 3 channels per image like rgb