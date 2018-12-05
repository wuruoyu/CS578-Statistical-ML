from __future__ import division
import math
import numpy as np
from decisiontree import DT_Classifier


class ThresholdLearner():
    def __init__(self):
        self.threshold_idx = None
        self.threshold_val = None
        self.sign = None
        self.error = None

    def fit(self, feature, label, distribution):
        size_feature = len(feature[0])
        for feature_idx in range(size_feature):
            min_val, max_val = self.find_feature_range(feature, feature_idx)
            for val in range(min_val, max_val):
                for sign in (True, False):
                    error = self.compute_error(
                        feature, label, distribution, feature_idx, val, sign)
                    if self.error == None or self.error > error:
                        self.threshold_idx = feature_idx
                        self.threshold_val = val
                        self.sign = sign
                        self.error = error
        print("idx: ", self.threshold_idx)
        print("val: ", self.threshold_val)
        print("sign: ", self.sign)
        print("error: ", self.error)
        return self.error

    def compute_error(self, feature, label, distribution, feature_idx, val, sign):
        size_sample = len(feature)
        error = 0
        for sample_idx in range(size_sample):
            flag = False
            if feature[sample_idx][feature_idx] > val:
                if label[sample_idx] == 1 and sign == False:
                    flag = True
                elif label[sample_idx] == -1 and sign == True:
                    flag = True
            else:
                if label[sample_idx] == 1 and sign == True:
                    flag = True
                elif label[sample_idx] == -1 and sign == False:
                    flag = True
            if flag:
                error += distribution[sample_idx]
        return error

    def find_feature_range(self, feature, feature_idx):
        feature_vals = [entry[feature_idx] for entry in feature]
        min_val = min(feature_vals)
        max_val = max(feature_vals)
        return min_val, max_val

    def predict(self, feature):
        if feature[self.threshold_idx] > self.threshold_val:
            if self.sign:
                return 1
            return -1
        else:
            if self.sign:
                return -1
            return 1


class Adaboost():
    def __init__(self, feature, label):
        assert(len(feature) == len(label))
        self.feature = feature
        self.label = label
        self.distribution = None
        self.weak_learner_list = []
        self.alphas = []

    def train(self, round):
        self.distribution = len(self.feature) * [1 / len(self.feature)]
        for _ in range(round):
            # fitting
            weak_learner = DT_Classifier(max_depth=1)
            weak_learner.fit(self.feature, self.label, self.distribution)

            # computing error
            error = 0
            predict_label = weak_learner.predict(self.feature)
            for idx in range(len(predict_label)):
                if predict_label[idx] != self.label[idx]:
                    error += self.distribution[idx]

            self.weak_learner_list.append(weak_learner)
            self.alphas.append(1/2 * np.log((1 - error) / error))
            self.redistribute()
            print("distribution", self.distribution)
            print("alpha: ", self.alphas[-1])
        print("training end")

    def predict(self, feature):
        output = [0] * 10
        for learner_idx in range(len(self.weak_learner_list)):
            output[self.weak_learner_list[learner_idx].predict(feature)] += self.alphas[learner_idx] 
        return output.index(max(output))

    def redistribute(self):
        temp_distribution = []
        count = 0
        # redistribute
        predict_label = self.predict(self.feature)
        for idx in range(len(self.feature)):
            if predict_label[idx] == self.label[idx]:
                count += 1
                temp_distribution.append(
                    self.distribution[idx] * math.e ** (-self.alphas[-1]))
            else:
                temp_distribution.append(
                    self.distribution[idx] * math.e ** (self.alphas[-1]))
        # normalize
        sum_temp = sum(temp_distribution)
        temp_distribution /= sum_temp

        self.distribution = temp_distribution
        print("accuracy: ", count / (len(self.feature)))


def main():
    data = np.genfromtxt('./winequality-white.csv',
                         delimiter=';', dtype=float, skip_header=1)
    train_feature = data[:3000, :-1]
    train_label = data[:3000, -1]

    test_feature = data[3000:, :-1]
    test_label = data[3000:, : -1]

    adaboost = Adaboost(train_feature, train_label)
    adaboost.train(100)


if __name__ == "__main__":
    main()
