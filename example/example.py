"""
Example usage of NeuralNetwork to solve the MNIST data set.
"""
import sys, os
import numpy

# Sloppily add neural_network to our path so we can import it
sys.path.insert(0, os.path.abspath('../neural_network'))

from neural_network import NeuralNet


def train_the_neural_net(neural_net, epochs=1):
    print 'Training the neural network.'
    training_data_file = open('mnist_train.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = epochs
    for i in range(epochs):
        print 'Training epoch {}/{}.'.format(i+1, epochs)
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99

            neural_net.train(inputs, targets)

    print 'complete.'


def test_the_neural_net(neural_net):
    print 'Testing the neural network.'
    test_data_file = open('mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []
    for i, record in enumerate(test_data_list):
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        outputs = neural_net.query(inputs)

        label = numpy.argmax(outputs)
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    print 'complete.'

    return scorecard


if __name__ == '__main__':

    print 'Starting neural network to recognize handwritten digits.'

    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1

    nn = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Train
    train_the_neural_net(nn, epochs=1)

    # Test
    test_results = numpy.asarray(test_the_neural_net(nn))

    # Print results
    print('Neural network is {}% accurate at predicting handwritten digits.'
        .format(test_results.sum() / float(test_results.size) * 100.0))
