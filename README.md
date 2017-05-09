
## Artificial Neural Network in Python

A simple 3-layer ANN (artificial neural network) written in Python.

Note: I have written this same 3-layer neural network in Go which you can find [here](https://github.com/michaelwayman/go-ann).


## Requirements
 - python 2.7 (I haven't tested any other version)
 - numpy
 - scipy


## The example

The example uses the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) to train and test the neural network.

The MNIST (Modified National Institute of Standards and Technology) database contains 60,000 training images and 10,000 testing images of handwritten numbers from 0-9.

#### Download

 - Training Data - [download](https://pjreddie.com/media/files/mnist_train.csv)
 - Testing Data - [download](https://pjreddie.com/media/files/mnist_test.csv)

#### Output

```
Starting neural network to recognize handwritten digits.
Training the neural network.
Training epoch 1/1.
complete.
Testing the neural network.
complete.
Neural network is 95.94% accurate at predicting handwritten digits.
```


## Performance

The example takes ~45s to complete.

**Note:** the same setup in *Go* can be found [here](https://github.com/michaelwayman/go-ann) and it **takes ~85s**. The numpy library has been so optimized when dealing with complex mathematics that it is hard to do better, even in a compiled language, even when coding for particular use cases.
