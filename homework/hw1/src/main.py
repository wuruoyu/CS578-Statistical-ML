from __future__ import division
import numpy as np

from metric import accuracy, precision, recall, f1
from k_cross import KFoldCross
from decisiontree import Node, DT_Classifier
from knn import KNN_Classifier

# for debug usage
from IPython import embed
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def test_metric():
    print("###precision###")
    print precision(test_label, knn_predict) 
    print precision_score(test_label, knn_predict, average='macro')

    print("###recall###")
    print recall(test_label, knn_predict) 
    print recall_score(test_label, knn_predict, average='macro')

    print("###f1###")
    print f1(test_label, knn_predict) 
    print f1_score(test_label, knn_predict, average='macro')


def test_kfold():
    knn = KNN_Classifier(k=1)


def main():
    my_data = np.genfromtxt('../winequality-white.csv', delimiter=';', dtype=float, skip_header=1)

    # proprocess(normalization and weighted vote)

    # train set
    train_feature = my_data[:3000, :-1]
    train_label = my_data[:3000, -1]

    # test set
    test_feature = my_data[3000:, :-1]
    test_label = my_data[3000:, -1]

    # feature and label
    feature = my_data[:, :-1]
    label = my_data[:, -1]

    embed()

    # test_metric()

    """
    # test knn
    knn = KNN_Classifier(k=1)
    knn.fit(train_feature, train_label)
    fuck = knn.predict(train_feature)
    print(accuracy(train_label, fuck))
    embed()
    return
    """

    # knn kfold
    """
    print("----KNN kfold with p = 1----")
    for i in range(1, 5):
        print("--k = ", i, "--")
        knn = KNN_Classifier(k=3, p=1)
        KFoldCross(knn, feature, label, 4) 
    
    print("##########################")
    print("----KNN kfold with p = 2----")
    for i in range(1, 5):
        print("--k = ", i, "--")
        knn = KNN_Classifier(k=3, p=2)
        KFoldCross(knn, feature, label, 4) 
    
    print("##########################")
    print("----KNN kfold with cosine----")
    for i in range(1, 5):
        print("--k = ", i, "--")
        knn = KNN_Classifier(k=3, metric='cosine')
        KFoldCross(knn, feature, label, 4) 
    """

    """
    # dt kfold
    print("----DT kfold----")
    for i in range(0, 7):
        print("--max depth = ", i, "--")
        dt = DT_Classifier(max_depth=i,sigmoid=True, min_impurity_decrease=1.0)
        KFoldCross(dt, feature, label, 4) 
    """

    # test knn
    # for i in range(5):
    #knn = KNN_Classifier(k=1)
    #knn.fit(train_feature, train_label)
    #knn_predict = knn.predict(test_feature)
    #print(accuracy(test_label, knn_predict))
    #print(precision(test_label, knn_predict))
    #print(recall(test_label, knn_predict))
    #print(f1(test_label, knn_predict))
   
    # test metric
    # test_metric()

    # test dt
    dt = DT_Classifier()
    dt.fit(train_feature, train_label)
    dt_predict = dt.predict(test_feature)
    print(accuracy_score(test_label, dt_predict))

    # use sklearn dt
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_feature, train_label)
    sklearn_predict = clf.predict(test_feature)
    print("sklearn dt accuracy: ", accuracy_score(test_label, sklearn_predict))
    print("sklearn dt f1: ", f1(test_label, sklearn_predict))

    # use sklearn knn
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_feature, train_label)
    sklearn_knn_predict = neigh.predict(test_feature)
    print("sklearn knn accuracy: ", accuracy_score(test_label, sklearn_knn_predict))
    print("sklearn knn f1: ", f1(test_label, sklearn_knn_predict))

if __name__ == "__main__":
    main()
