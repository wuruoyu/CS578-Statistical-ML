from __future__ import division
import math
import numpy as np

class ThresholdLearner():
    def __init__(self):
        self.threshold_idx = None
        self.threshold_val = None
        self.sign = None
        self.error = None

    def fit(self, feature, label, distribution):
        size_feature = len(feature[0])
        for feature_idx in range(size_feature):
            min_val, max_val = self.find_feature_range(feature ,feature_idx)
            for val in range(min_val, max_val):
                for sign in (True, False):
                    error = self.compute_error(feature, label, distribution, feature_idx, val, sign) 
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

class DecisionStump():
    def __init__(self):
        pass

    def fit(self):
        pass

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
            weak_learner = ThresholdLearner()
            error = weak_learner.fit(self.feature, self.label, self.distribution)
            self.weak_learner_list.append(weak_learner)
            self.alphas.append(1/2 * np.log((1 - error) / error))
            self.redistribute()
            print("distribution", self.distribution)
            print("alpha: ", self.alphas[-1])

    def predict(self, item):
        output = 0
        for learner_idx in range(len(self.weak_learner_list)):
            output += self.alphas[learner_idx] * self.weak_learner_list[learner_idx].predict(item)
        if output > 0:
            return 1
        else:
            return -1

    def redistribute(self):
        temp_distribution = []
        count = 0
        # redistribute
        for idx in range(len(self.feature)):
            if self.predict(self.feature[idx]) == self.label[idx]:
                count += 1
                temp_distribution.append(self.distribution[idx] * math.e ** (-self.alphas[-1]))
            else:
                temp_distribution.append(self.distribution[idx] * math.e ** (self.alphas[-1]))
        # normalize
        sum_temp = sum(temp_distribution)
        temp_distribution /= sum_temp

        self.distribution = temp_distribution
        print("accuracy: ", count / (len(self.feature)))

def main():
    data = np.genfromtxt(
        # './winequality-white.csv', delimiter=';', dtype=float, skip_header=1)
    feature = data[:, :-1]
    label = data[:, -1]

    adaboost = Adaboost(feature, label)
    adaboost.train(2)

if __name__ == "__main__":
    main()
