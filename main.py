import numpy as np
import matplotlib.pyplot as plt
from dataset import Dataset


def expit(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate
        self.activation_function = lambda x: expit(x)

    def train(self, inputs_list, targets_list):
        inputd = np.array(inputs_list, ndmin=2).T
        targetd = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputd)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targetd - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputd))

    def query(self, inputs_list):
        inputd = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputd)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


accuracies = []
symbols = ['Chevron', 'EinSieben', 'Kreis', 'Kreuz', 'Strich']

input_nodes = 625
hidden_nodes = 20
output_nodes = 5
learning_rate = .4
epochs = 15
split = 0.95

network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
dataset = Dataset(split)
training_data_list, test_data_list = dataset.load_data()


def plotImage(inputs):
    plt.imshow(np.asfarray(inputs).reshape(25, 25), cmap='Greys')
    plt.show()


def test(data):
    scorecard = []
    for record in data:
        correct_label = int(record[0])
        inputs = (np.asfarray(record[1:]) / 255.0 * 0.99) + 0.01
        outputs = network.query(inputs)
        label = np.argmax(outputs)
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    scorecard_array = np.asarray(scorecard)
    accuracies.append(scorecard_array.sum() / scorecard_array.size * 100)


def start():
    for e in range(epochs):
        for record in training_data_list:
            inputs = (np.asfarray(record[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(record[0])] = 0.99
            network.train(inputs, targets)
        test(training_data_list)
        print("Epoch {0}  accuracy: {1:.2f}%".format(e + 1, accuracies[-1]))


if __name__ == '__main__':
    start()
    plt.plot(accuracies)
    plt.title('Accuracy: {:.2f}%'.format(accuracies[-1]))
    plt.show()
