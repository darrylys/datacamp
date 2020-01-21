
import random
import numpy as np
from perceptron import Perceptron
from linear_regression import LinearRegression
from commons import generateRandomDataset

class CoinSimulation:
    def __init__(self, ncoins=1000, nflips=10):
        self.ncoins = ncoins
        self.nflips = nflips
        self.v1 = 0
        self.vmin = 0
        self.vrand = 0

    def __flipcoins(self):
        nheads = 0

        # rather than generating random every 10 times, just generate
        # random 10 bits at once! This makes the function considerably faster
        glob = random.randint(0, 2**self.nflips)
        while glob > 0:
            if glob % 2 == 1:
                nheads += 1
            glob = glob // 2

        return nheads

    def run(self):
        cheads = [self.__flipcoins() for x in range(0, self.ncoins)]
        self.v1 = cheads[0] / self.nflips
        self.vrand = cheads[random.randint(0, len(cheads)-1)] / self.nflips
        self.vmin = min(cheads) / self.nflips
        
def simulateCoinFlips():
    i = 0
    reps = 100000

    sumVmin = 0.0
    sumVrand = 0.0
    sumV1 = 0.0

    while i < reps:
        simulation = CoinSimulation()
        simulation.run()

        sumV1 += simulation.v1
        sumVrand += simulation.vrand
        sumVmin += simulation.vmin

        i += 1

        if i % 10000 == 0:
            print("%d" % i)
    
    print("avg Vmin: %f, avg Vrand: %f, avg V1: %f" % (
        sumVmin / reps, sumVrand / reps, sumV1 / reps
    ))

def Q_1_2():
    simulateCoinFlips()

def Q_5_6(N=100, totalIterations=1000, Ntest=1000):
    _I = 0
    sumEin = 0
    sumEout = 0

    while _I < totalIterations:
        X, y, m, c = generateRandomDataset(N)
        linreg = LinearRegression()
        linreg.train(X, y)
        sumEin += linreg._ein
        Xtest, ytest, m, c = generateRandomDataset(Ntest, m, c)
        sumEout += linreg.test(Xtest, ytest)
        _I += 1

    print("avg Ein: %f" % (sumEin / totalIterations))
    print("avg Eout: %f" % (sumEout / totalIterations))

def Q_7(N=10, totalIterations=1000):
    _I = 0
    sumPerceptronIters = 0

    while _I < totalIterations:
        X, y, m, c = generateRandomDataset(N)
        linreg = LinearRegression()
        linreg.train(X, y)
        pla = Perceptron()
        pla.train(X, y, initW=linreg._w)
        sumPerceptronIters += pla._numIterations
        _I += 1

    print("avg Perceptron number of iterations: %f" % (sumPerceptronIters / totalIterations))


def main():
    Q_5_6()
    Q_7()
    
if __name__ == '__main__':
    main()
