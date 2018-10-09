import sys
import numpy as np
import struct
import gzip
from random import shuffle

from IPython import embed


class VanillaPerceptron():
    def __init__(self, train_data, train_label, learning_rate, epoch):
        self.train_data = train_data
        self.train_label = train_label
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.weights = [0.0] * self.train_data.shape[1] * self.train_data.shape[2]
        # mnist specific
        assert len(self.weights) = 28 * 28
        self.bias = 0

    def train(self):
        for _ in range(self.epoch):
            for idx in range(len(self.train_data)):
                data = self.train_data[idx]
                label = self.train_label[idx]
                activate = np.dot(self.weights, data) + self.bias
                if label * activate < 0:
                    for w_idx in range(len(self.weights)):
                        self.weights[w_idx] += self.learning_rate * label * data[w_idx]
                    self.bias += self.learning_rate * label


    def predict(self, data):
        activate = np.dot(self.weights, data) + self.bias
        if activate > 0:
            return True
        return False


def read_mnist(filename):
    with gzip.open(filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def main(argv):
    size_training = int(argv[0])
    epoch = int(argv[1])
    learning_rate = float(argv[2])
    path = argv[3]

    training_img_path = path + '/train-images-idx3-ubyte.gz'
    training_label_path = path + '/train-labels-idx1-ubyte.gz'
    test_img_path = path + '/t10k-images-idx3-ubyte.gz'
    test_label_path = path + '/t10k-labels-idx1-ubyte.gz'

    training_img = read_mnist(training_img_path)[:size_training]
    training_label = read_mnist(training_label_path)[:size_training]

    # TODO: shuffle the training set and transform it into 1-d list
    shuffle()

    # build 10 perceptron, one classifier for one digit
    perceptron_clfs = []
    for target_label in range(10):
        labels = []
        for label in training_label:
            if label == target_label:
                labels.append(1)
            else:
                labels.append(-1)
        perceptron_clfs.append(
            VanillaPerceptron(training_img, labels, learning_rate, epoch))

    # training
    for clf in perceptron_clfs:
        clf.train()

    # testing and scoring

if __name__ == "__main__":
    main(sys.argv[1:])
