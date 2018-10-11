from __future__ import division
import sys
import numpy as np
import struct
import gzip
import random
from metric import f1, accuracy

from IPython import embed
from multiprocessing import Pool

class Winnow():
    def __init__(self, training_set, factor, epoch):
        self.training_set = training_set
        self.factor = factor 
        self.epoch = epoch

        # mnist specific
        self.weights = [1.0] * 28 * 28

    def train(self):
        for _ in range(self.epoch):
            for idx in range(len(self.training_set)):
                data = self.training_set[idx][0]
                label = self.training_set[idx][1]
                activate = np.dot(self.weights, data)
                if activate < 1 and label > 0:
                    for w_idx in range(len(self.weights)):
                        if data[w_idx] > 0:
                            self.weights[w_idx] *= self.factor 
                if activate >= 1 and label < 0:
                    for w_idx in range(len(self.weights)):
                        if data[w_idx] > 0:
                            self.weights[w_idx] /= self.factor 

    def predict(self, data):
        activate = np.dot(self.weights, data) 
        return activate 


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
    testing_img_path = path + '/t10k-images-idx3-ubyte.gz'
    testing_label_path = path + '/t10k-labels-idx1-ubyte.gz'

    # read the file, trans to 1-d array
    training_img = read_mnist(training_img_path)[:size_training].reshape(
        size_training, 784)
    training_label = read_mnist(training_label_path)[:size_training]
    testing_img = read_mnist(testing_img_path)
    testing_img = testing_img.reshape(len(testing_img), 28 * 28)
    testing_label = read_mnist(testing_label_path)

    # shuffle(I dont like it)
    training_set = zip(training_img, training_label)
    random.shuffle(training_set)
    training_img, training_label = zip(*training_set)

    # scale [0, 255] to boolean
    training_img = [[round(pixel/255) for pixel in sample] for sample in training_img]
    testing_img = [[round(pixel/255) for pixel in sample] for sample in testing_img]

    # build 10 perceptron, one classifier for one digit
    perceptron_clfs = []
    for target_label in range(10):
        local_training_label = [
            1 if label == target_label else -1 for label in training_label
        ]
        # pack together and shuffle
        local_training_set = zip(training_img, local_training_label)
        perceptron_clfs.append(
            Winnow(local_training_set, learning_rate, epoch))

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
    train_f1 = f1(training_label, predict_label)

    # test on testing set
    testing_set = zip(testing_img, testing_label)
    predict_label = []
    for item in testing_set:
        # predict
        scores = []
        for clf in perceptron_clfs:
            scores.append(clf.predict(item[0]))
        predict_label.append(scores.index(max(scores)))
    test_f1 = f1(testing_label, predict_label)

    return (train_f1, test_f1)

def plot():
    argvs = []
    for size in range(500, 10250, 250):
        argv = []
        argv.append(size)
        argv.append(20)
        argv.append(1.2)
        argv.append('data')
        argvs.append(argv)

    process_pool = Pool(10)
    return process_pool.map(main, argvs)


if __name__ == "__main__":
    output = plot()
    import matplotlib.pyplot as plt
    train_f1 = [item[0] for item in output]
    test_f1 = [item[1] for item in output]
    plt.plot(range(500, 10250, 250), train_f1, 'g--', label="train")
    plt.plot(range(500, 10250, 250), test_f1, 'r--', label="test")
    plt.axis([500, 10000, 0.6, 1])
    plt.show()
    embed()

