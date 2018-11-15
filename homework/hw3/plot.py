import matplotlib.pyplot as plt

def plot(acc_train, acc_test):
    assert(len(acc_train) == len(acc_test))
    epoch = len(acc_train)
    plt.plot(range(epoch), acc_train, 'b--')
    plt.plot(range(epoch), acc_test, 'g')
    plt.show()
