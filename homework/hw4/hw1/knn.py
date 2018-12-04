from __future__ import division
import numpy as np
from math import sqrt

from metric import accuracy, f1, sigmoid
from k_cross import KFoldCross

# debug
from IPython import embed

class KNN_Classifier:
    def __init__(self, k=1, metric='minkowski', p=2, weights='uniform'):
        self.X = None
        self.y = None
        self.k = k
        self.metric = metric
        self.p = p
        self.weights = weights
        self.print_params()

    def print_params(self):
        print "Hyperparameters: "
        print("K: ", self.k)
        if self.metric == 'minkowski':
            print("Distance mesure: Minkowski distance with p =", self.p)
        elif self.metric == 'cosine':
            print("Distance mesure: Cosine distance")
        print

    def fit(self, X, y):
        """
        learn from traning set

        X : features

        y : target
        """
        self.X = X
        self.y = y

    def compute_distance(self, x, y):
        if len(x) != len(y):
            raise ValueError('len should be same')

        if self.metric == 'minkowski':
            dist = 0
            for idx in range(len(x)):
                dist += abs(x[idx] - y[idx])**self.p
            return dist**(1 / self.p)
        elif self.metric == 'cosine':
            return 1 - np.dot(x, y) / (sqrt(np.dot(x, x)) * sqrt(np.dot(y, y)))
        else:
            raise ValueError('No such metric')

    def major_label(self, labels):
        return max(labels, key=labels.count)

    def predict(self, input):
        """
        predit on test set
        """
        return_label = []
        for to_predict in input:
            k_n_n = []
            for data_idx in range(len(self.X)):
                k_n_n.append((self.compute_distance(to_predict, self.X[data_idx]),
                              data_idx))
            k_n_n.sort(key=lambda pair: pair[0])
            labels = []
            for idx in range(self.k):
                labels.append(self.y[k_n_n[idx][1]])
            return_label.append(self.major_label(labels))

        return return_label

    def reset(self):
        """
        reset the model
        """
        pass


def main():
    data = np.genfromtxt(
        '../winequality-white.csv', delimiter=';', dtype=float, skip_header=1)
    feature = data[:, :-1]
    label = data[:, -1]

    knn = KNN_Classifier(k=6, p=1)
    KFoldCross(knn, feature, label, 4)

if __name__ == "__main__":
    main()
