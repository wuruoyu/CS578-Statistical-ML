from __future__ import division
import sys
import numpy as np
import struct
import gzip
import random
from metric import f1, accuracy

from IPython import embed


class VanillaPerceptron():
    def __init__(self, training_set, learning_rate, epoch):
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.epoch = epoch

        # mnist specific
        self.weights = [0.0] * 28 * 28
        self.bias = 0

    def train(self):
        for _ in range(self.epoch):
            for idx in range(len(self.training_set)):
                data = self.training_set[idx][0]
                label = self.training_set[idx][1]
                activate = np.dot(self.weights, data) + self.bias
                if label * activate <= 0:
                    for w_idx in range(len(self.weights)):
                        self.weights[
                            w_idx] += self.learning_rate * label * data[w_idx]
                    self.bias += self.learning_rate * label

    def predict(self, data):
        activate = np.dot(self.weights, data) + self.bias
        return activate


def read_mnist(filename):
    with gzip.open(filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def max_pooling(img):
    assert(len(img) == 28 * 28)
    return_img = []
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            print i, j
            return_img.append(max([img[28 * i + j], img[28 * i + j + 1],
                                   img[28 * (i + 1) + j], img[28 * (i + 1) + j + 1]]))
    embed()
    return return_img


def main(argv):
    is_regularization = bool(argv[0])
    type_feature = argv[1]
    path = argv[2]
    size_training = 10000

    training_img_path = path + '/train-images-idx3-ubyte.gz'
    training_label_path = path + '/train-labels-idx1-ubyte.gz'
    testing_img_path = path + '/t10k-images-idx3-ubyte.gz'
    testing_label_path = path + '/t10k-labels-idx1-ubyte.gz'

    # read the file, trans to 1-d array
    training_img = read_mnist(training_img_path)[:size_training].reshape(
        size_training, 784)
    training_label = read_mnist(training_label_path)[:size_training]
    testing_img = read_mnist(testing_img_path)
    testing_img = testing_img.reshape(len(testing_img), 28 * 28)
    testing_label = read_mnist(testing_label_path)

    # shuffle
    training_set = zip(training_img, training_label)
    random.shuffle(training_set)
    training_img, training_label = zip(*training_set)

    if type_feature == 'type1':
        training_img = [[pixel/255 for pixel in item] for item in training_img]
        testing_img = [[pixel/255 for pixel in item] for item in testing_img]
    elif type_feature == 'type2':
        training_img = [max_pooling(item) for item in training_img]
        testing_img = [max_pooling(item) for item in testing_img]
    else:
        raise ValueError

    # add bias term
    for sample in training_img:
        sample.append(1)
    for sample in testing_img:
        sample.append(1)

    # build 10 perceptron, one classifier for one digit
    perceptron_clfs = []
    for target_label in range(10):
        local_training_label = [
            1 if label == target_label else -1 for label in training_label
        ]
        # pack together and shuffle
        local_training_set = zip(training_img, local_training_label)
        perceptron_clfs.append(
            VanillaPerceptron(local_training_set, learning_rate, epoch))

    # training
    for clf in perceptron_clfs:
        clf.train()

    # test on training set
    training_set = zip(training_img, training_label)
    predict_label = []
    for item in training_set:
        # predict
        scores = []
        for clf in perceptron_clfs:
            scores.append(clf.predict(item[0]))
        predict_label.append(scores.index(max(scores)))
    print("Train F1 score: ", f1(training_label, predict_label))
    print("Train accuracy: ", accuracy(training_label, predict_label))

    # test on testing set
    testing_set = zip(testing_img, testing_label)
    predict_label = []
    for item in testing_set:
        # predict
        scores = []
        for clf in perceptron_clfs:
            scores.append(clf.predict(item[0]))
        predict_label.append(scores.index(max(scores)))
    print("Test F1 score: ", f1(testing_label, predict_label))
    print("Test accuracy: ", accuracy(testing_label, predict_label))


if __name__ == "__main__":
    main(sys.argv[1:])
