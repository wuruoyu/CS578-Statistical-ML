from __future__ import division
import sys
import numpy as np
import struct
import gzip
import random
from metric import f1, accuracy, sigmoid, mse
from math import exp

from IPython import embed


class BGD():
    def __init__(self, training_set):
        self.training_set = training_set

        # TODO: hyper-parameter
        self.learning_rate = 0.01
        self.regularization_rate = 0.0001
        self.stop_step = 5

        self.weights = [0.0] * (len(training_set[0][0]))

    def plot(self):
        pass

    def train(self):
        w_gradient = [0.0] * len(self.weights)
        # sum up gradient
        for idx in range(len(self.training_set)):
            data = self.training_set[idx][0]
            label = self.training_set[idx][1]
            for w_gradient_idx in range(len(w_gradient)):
                w_gradient[w_gradient_idx] += (sigmoid(
                    np.dot(self.weights, data)) - label) * data[w_gradient_idx]
        # update weights
        for w_gradient_idx in range(len(w_gradient)):
            w_gradient[w_gradient_idx] /= len(self.training_set)
            w_gradient[w_gradient_idx] += self.regularization_rate * \
                self.weights[w_gradient_idx]
            self.weights[w_gradient_idx] -= self.learning_rate * \
                w_gradient[w_gradient_idx]

    def predict(self, data):
        return sigmoid(np.dot(self.weights, data))

    def training_loss(self):
        pass


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
            return_img.append(max([img[28 * i + j], img[28 * i + j + 1],
                                   img[28 * (i + 1) + j], img[28 * (i + 1) + j + 1]]))
    assert(len(return_img) == 14 * 14)
    return return_img


def main(argv):
    is_regularization = bool(argv[0])
    type_feature = argv[1]
    path = argv[2]
    # use first 10000 sample
    size_training = 8000
    size_validation = 2000
    max_step = 500
    stop_step = 5

    training_img_path = path + '/train-images-idx3-ubyte.gz'
    training_label_path = path + '/train-labels-idx1-ubyte.gz'
    testing_img_path = path + '/t10k-images-idx3-ubyte.gz'
    testing_label_path = path + '/t10k-labels-idx1-ubyte.gz'

    # read the file, trans to 1-d array
    training_img = read_mnist(training_img_path)[:size_training].reshape(
        size_training, 784)
    training_label = read_mnist(training_label_path)[:size_training]
    validation_img = read_mnist(training_img_path)[
        size_training: size_training + size_validation].reshape(size_validation, 784)
    validation_label = read_mnist(training_img_path)[
        size_training:size_training + size_validation]
    testing_img = read_mnist(testing_img_path)
    testing_img = testing_img.reshape(len(testing_img), 28 * 28)
    testing_label = read_mnist(testing_label_path)

    # shuffle
    training_set = zip(training_img, training_label)
    random.shuffle(training_set)
    training_img, training_label = zip(*training_set)

    if type_feature == 'type1':
        training_img = [[pixel/255 for pixel in item] for item in training_img]
        validation_img = [[pixel/255 for pixel in item] for item in training_img]
        testing_img = [[pixel/255 for pixel in item] for item in testing_img]
    elif type_feature == 'type2':
        training_img = [max_pooling(item) for item in training_img]
        validation_img = [max_pooling(item) for item in training_img]
        testing_img = [max_pooling(item) for item in testing_img]
    else:
        raise ValueError

    # add bias term
    for sample in training_img:
        sample.append(1)
    for sample in validation_img:
        sample.append(1)
    for sample in testing_img:
        sample.append(1)

    # build 10 classifier, one classifier for one digit
    clfs = []
    for target_label in range(10):
        local_training_label = [
            1 if label == target_label else 0 for label in training_label
        ]
        # pack together
        local_training_set = zip(training_img, local_training_label)
        clfs.append(BGD(local_training_set))

    # training
    acc_on_validation = []
    testing_set = zip(testing_img, testing_label)
    for step in range(max_step):
        # stop criteria
        if step >= stop_step:
            if acc_on_validation[-1] <= max(self.acc_on_validation[-1-self.stop_step: -1]):
                break
        for clf in clfs:
            clf.train()

        # compute training loss
        for item in training_set:
            for clf in clfs:
                pass

        # see acc on training set
        training_set = zip(training_img, training_label)
        predict_label = []
        for item in training_set:
            scores = []
            for clf in clfs:
                scores.append(clf.predict(item[0]))
            predict_label.append(scores.index(max(scores)))
        training_acc = accuracy(training_label, predict_label)
        
        # see acc on testing set
        predict_label = []
        for item in testing_set:
            scores = []
            for clf in clfs:
                scores.append(clf.predict(item[0]))
            predict_label.append(scores.index(max(scores)))
        testing_acc = accuracy(testing_label, predict_label)

        # print
        print("epoch ", step, ": ", "Training loss: ", )

    # test on testing set
    testing_set = zip(testing_img, testing_label)
    predict_label = []
    for item in testing_set:
        # predict
        scores = []
        for clf in clfs:
            scores.append(clf.predict(item[0]))
        predict_label.append(scores.index(max(scores)))
    print("Test F1 score: ", f1(testing_label, predict_label))
    print("Test accuracy: ", accuracy(testing_label, predict_label))


if __name__ == "__main__":
    main(sys.argv[1:])
