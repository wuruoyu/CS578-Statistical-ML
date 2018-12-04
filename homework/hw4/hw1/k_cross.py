from __future__ import division
import numpy as np
from metric import precision, recall, f1, accuracy


def KFoldCross(model, feature=None, label=None, cv=4):
    """
    data : array-like, data to fit

    target : array-like, target variable

    cv : int, 
    """
    assert len(feature) == len(label)
    num_test_sample = len(feature) // cv
    # num_train_sample = num_train_train_sample + num_validation_sample
    num_train_sample = len(feature) - num_test_sample
    num_validation_sample = num_train_sample // cv
    num_train_train_sample = num_train_sample - num_validation_sample
    assert num_test_sample + num_train_train_sample + num_validation_sample == len(
        feature)

    total_train_f1 = 0
    total_train_accuracy = 0
    total_validation_f1 = 0
    total_validation_accuracy = 0
    total_test_f1 = 0
    total_test_accuracy = 0

    # test fold
    for test_fold in range(0, cv):
        # split test and train
        test_sample_feature = feature[test_fold * num_test_sample:
                                      (test_fold + 1) * num_test_sample]
        test_sample_label = label[test_fold * num_test_sample:(test_fold + 1) *
                                  num_test_sample]
        train_sample_feature = np.concatenate(
            (feature[:test_fold * num_test_sample],
             feature[(test_fold + 1) * num_test_sample:]),
            axis=0)
        train_sample_label = np.concatenate(
            (label[:test_fold * num_test_sample],
             label[(test_fold + 1) * num_test_sample:]),
            axis=0)
        # check
        assert len(test_sample_feature) == num_test_sample == len(
            test_sample_label)
        assert len(train_sample_feature) == num_train_sample == len(
            train_sample_label)

        total_fold_train_f1 = 0
        total_fold_train_accuracy = 0
        total_fold_validation_f1 = 0
        total_fold_validation_accuracy = 0

        # train fold
        for validation_fold in range(0, cv):
            model.reset()

            # split train and validation
            validation_sample_feature = train_sample_feature[
                validation_fold * num_validation_sample:(validation_fold + 1) *
                num_validation_sample]
            validation_sample_label = train_sample_label[
                validation_fold * num_validation_sample:(validation_fold + 1) *
                num_validation_sample]
            train_train_sample_feature = np.concatenate(
                (train_sample_feature[:validation_fold *
                                      num_validation_sample],
                 train_sample_feature[(validation_fold + 1) *
                                      num_validation_sample:]),
                axis=0)
            train_train_sample_label = np.concatenate(
                (train_sample_label[:validation_fold * num_validation_sample],
                 train_sample_label[(validation_fold + 1) *
                                    num_validation_sample:]),
                axis=0)
            assert len(validation_sample_label) == len(
                validation_sample_feature) == num_validation_sample
            assert len(train_train_sample_feature) == len(
                train_train_sample_label) == num_train_train_sample

            model.fit(train_train_sample_feature, train_train_sample_label)

            # train stat
            train_output = model.predict(train_train_sample_feature)
            total_train_f1 += f1(train_train_sample_label, train_output)
            total_train_accuracy += accuracy(train_train_sample_label,
                                             train_output)
            total_fold_train_f1 += f1(train_train_sample_label, train_output)
            total_fold_train_accuracy += accuracy(train_train_sample_label,
                                                  train_output)

            # validation stat
            validation_output = model.predict(validation_sample_feature)
            total_validation_f1 += f1(validation_sample_label,
                                      validation_output)
            total_validation_accuracy += accuracy(validation_sample_label,
                                                  validation_output)
            total_fold_validation_f1 += f1(validation_sample_label,
                                           validation_output)
            total_fold_validation_accuracy += accuracy(validation_sample_label,
                                                       validation_output)

        # predict in test set
        model.fit(train_sample_feature, train_sample_label)
        output = model.predict(test_sample_feature)

        # fold stat(overall)
        total_test_f1 += f1(test_sample_label, output)
        total_test_accuracy += accuracy(test_sample_label, output)

        # fold statistics(local)
        fold_train_f1 = total_fold_train_f1 / cv
        fold_train_accuracy = total_fold_train_accuracy / cv
        fold_validation_f1 = total_fold_validation_f1 / cv
        fold_validation_accuracy = total_fold_validation_accuracy / cv
        fold_test_f1 = f1(test_sample_label, output)
        fold_test_accuracy = accuracy(test_sample_label, output)

        print("Fold-", test_fold + 1)
        print("Training: F1 score: ", fold_train_f1, ", Accuracy: ",
              fold_train_accuracy)
        print("Validation: F1 score: ", fold_validation_f1, ", Accuracy: ",
              fold_validation_accuracy)
        print("Testing: F1 score: ", fold_test_f1, ", Accuracy: ",
              fold_test_accuracy)
        print

    # statistics
    train_f1 = total_train_f1 / (cv * cv)
    train_accuracy = total_train_accuracy / (cv * cv)
    validation_f1 = total_validation_f1 / (cv * cv)
    validation_accuracy = total_validation_accuracy / (cv * cv)
    test_f1 = total_test_f1 / cv
    test_accuracy = total_test_accuracy / cv

    print("Average")
    overall_report_line = [
        train_f1, train_accuracy, validation_f1, validation_accuracy, test_f1,
        test_accuracy, '\n'
    ]
    print("Training: F1 score: ", train_f1, ", Accuracy: ", train_accuracy)
    print("Validation: F1 score: ", validation_f1, ", Accuracy: ",
          validation_accuracy)
    print("Testing: F1 score: ", test_f1, ", Accuracy: ", test_accuracy)
    print
