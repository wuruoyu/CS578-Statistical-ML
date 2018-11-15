from multiprocessing import Pool

def mul_train(clf):
    clf.train()

class test:
    def __init__(self):
        self.step = 0

    def train(self):
        self.step += 1
        print(self.step)

p = Pool(10)
clfs = []
for i in range(10):
    clfs.append(test())

p.map(mul_train, clfs)
p.map(mul_train, clfs)
