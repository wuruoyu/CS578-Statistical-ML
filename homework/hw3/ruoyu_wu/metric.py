from __future__ import division
import math


def accuracy(target, predict):
    count = 0
    for i in range(len(predict)):
        if predict[i] == target[i]:
            count += 1
    return count / len(predict)


def sub_precision(target, predict, label):
    tp = 0
    fp = 0
    for idx in range(len(predict)):
        if predict[idx] == label:
            if target[idx] == label:
                tp += 1
            else:
                fp += 1
    return tp / (fp + tp)


def sub_recall(target, predict, label):
    tp = 0
    fn = 0
    for idx in range(len(predict)):
        if predict[idx] == label and target[idx] == label:
            tp += 1
        if predict[idx] != label and target[idx] == label:
            fn += 1
    if fn + tp == 0:
        return 0
    return tp / (fn + tp)


def precision(target, predict):
    predict_label_list = list(set(predict))
    total_precision = 0
    for label in predict_label_list:
        total_precision += sub_precision(target, predict, label)
    return total_precision / len(predict_label_list)


def recall(target, predict):
    predict_label_list = list(set(predict))
    total_recall = 0
    for label in predict_label_list:
        total_recall += sub_recall(target, predict, label)
    return total_recall / len(predict_label_list)


def f1(target, predict):
    predict_label_list = list(set(predict))
    total_f1 = 0
    for predict_label in predict_label_list:
        p = sub_precision(target, predict, predict_label)
        r = sub_recall(target, predict, predict_label)
        if p + r > 0:
            total_f1 += 2 * p * r / (p + r)
    return total_f1 / len(predict_label_list)


def sigmoid(x):
    try:
        output = 1 / (1 + math.exp(-x))
    except OverflowError:
        output = 0.0000001
    if output == 1:
        output = 0.9999999
    return output

def mse(target, predict):
    assert(len(target) == len(predict))
    output = 0
    for idx in range(len(target)):
        output += (target[idx] - predict[idx]) ** 2
    return output / len(target)
