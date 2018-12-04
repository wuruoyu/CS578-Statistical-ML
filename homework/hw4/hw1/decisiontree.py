from __future__ import division
import numpy as np
import math

from metric import sigmoid
from k_cross import KFoldCross


class Node:
    def __init__(self):
        self.data_idx = []
        self.split_idx = None
        # for continuous attribute
        self.split_threshold = None
        self.left_child = None
        self.right_child = None
        self.is_leaf = False
        self.classification = None


class DT_Classifier:
    """
    consider all attributes as continuous

    each node can only have two child, i.e. left and right
    left_child contains the point with ... lower

    when searching for split point, the range will be local

    split_attempt: number of attempts to find the best threshold
    """

    def __init__(self,
                 max_depth=5,
                 min_samples_split=2,
                 min_impurity_decrease=0,
                 split_attempt=10,
                 sigmoid=False):
        self.X = None
        self.y = None
        self.root = Node()
        self.feature_size = None
        self.label = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.split_attempt = split_attempt
        self.sigmoid = sigmoid
        self.print_param()

    def print_param(self):
        print("Hyper-Parameters:")
        print("max_depth: ", self.max_depth + 1)
        print("min_samples_split: ", self.min_samples_split)
        print("min_impurity_decrease: ", self.min_impurity_decrease)
        print("split_attempt: ", self.split_attempt)
        print("sigmoid normalization: ", self.sigmoid)

    def fit(self, X, y):
        """
        learn from traning set

        X : features

        y : target
        """
        self.X = X
        self.y = y
        self.feature_size = len(X[0])
        self.label = list(set(self.y))
        # assign idx
        self.root.data_idx = list(range(len(X)))
        self.build(self.root, 0)

    def major_label(self, node):
        """
        return the major label
        """
        label_counter = [0] * len(self.label)
        for idx in node.data_idx:
            label_counter[self.label.index(self.y[idx])] += 1
        # embed()
        return self.label[label_counter.index(max(label_counter))]

    def find_feature_range(self, node, feature_idx):
        """
        return the range of feature(locally)
        """
        feature_vals = [self.X[idx][feature_idx] for idx in node.data_idx]
        min_val = min(feature_vals)
        max_val = max(feature_vals)
        return min_val, max_val

    def compute_entropy(self, node):
        """
        return the entropy
        """
        label_counter = [0] * len(self.label)
        for idx in node.data_idx:
            label_counter[self.label.index(self.y[idx])] += 1

        entropy = 0
        for label_count in label_counter:
            if label_count != 0:
                p = label_count / len(node.data_idx)
                entropy -= p * math.log(p, 2)
        return entropy

    def branch(self, node, feature_idx, split_threshold):
        """
        return two split node
        """
        left_child = Node()
        right_child = Node()
        if self.sigmoid:
            for idx in node.data_idx:
                if sigmoid(self.X[idx][feature_idx]) <= split_threshold:
                    left_child.data_idx.append(idx)
                else:
                    right_child.data_idx.append(idx)
        else:
            for idx in node.data_idx:
                if self.X[idx][feature_idx] <= split_threshold:
                    left_child.data_idx.append(idx)
                else:
                    right_child.data_idx.append(idx)
        return left_child, right_child

    def choose_feature_and_branch(self, node):
        """
        assign (left_child, right_child) to node.child  
        except impurity_decrease < min_impurity_decrease
        """
        best_information_gain = 0
        best_left_child = None
        best_right_child = None
        best_split_idx = 0
        best_split_threshold = None

        # split feature
        for feature_idx in range(self.feature_size):
            if self.sigmoid:
                split_threshold_candidate = np.arange(0.1, 1,
                                                      1 / self.split_attempt)
            else:
                min_val, max_val = self.find_feature_range(node, feature_idx)
                if max_val > min_val:
                    split_threshold_candidate = np.arange(
                        min_val, max_val,
                        (max_val - min_val) / self.split_attempt)
                else:
                    continue
            # split threshold
            for val in split_threshold_candidate:
                left_child, right_child = self.branch(node, feature_idx, val)
                information_gain = -self.compute_entropy(
                    node) + self.compute_entropy(
                        left_child) + self.compute_entropy(right_child)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_left_child = left_child
                    best_right_child = right_child
                    best_split_idx = feature_idx
                    best_split_threshold = val

        if best_information_gain < self.min_impurity_decrease:
            return
        node.left_child = best_left_child
        node.right_child = best_right_child
        node.split_idx = best_split_idx
        node.split_threshold = best_split_threshold

    def build(self, node, height):
        major_label = self.major_label(node)

        # if growing to max_depth or number of samples below min_samples_split
        if height == self.max_depth or len(
                node.data_idx) < self.min_samples_split:
            node.is_leaf = True
            node.classification = major_label
            return

        self.choose_feature_and_branch(node)

        # if the gain < threshold
        if node.left_child == None and node.right_child == None:
            node.is_leaf = True
            node.classification = major_label
            return

        self.build(node.left_child, height + 1)
        self.build(node.right_child, height + 1)
        return

    def predict(self, X):
        """
        return the predict label
        """
        y = []
        for to_predict in X:
            node = self.root
            while not node.is_leaf:
                if to_predict[node.split_idx] < node.split_threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            y.append(node.classification)
        return y

    def reset(self):
        """
        reset the model
        """
        self.root = Node()
        self.feature_size = None
        self.label = None

    def prune(self):
        """
        prue the tree to generalize
        """
        pass


def main():
    data = np.genfromtxt(
        '../winequality-white.csv', delimiter=';', dtype=float, skip_header=1)
    feature = data[:, :-1]
    label = data[:, -1]

    dt = DT_Classifier(max_depth=1, sigmoid=True)
    KFoldCross(dt, feature, label, 4)


if __name__ == "__main__":
    main()
