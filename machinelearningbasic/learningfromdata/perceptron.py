
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

class Perceptron:
    def __init__(self):
        self._plaW = []
        self._numIterations = 0

    def __updatePLAWeights(self, plaW, yn, xn):
        i = 0
        pl = len(plaW)

        while i < pl:
            plaW[i] += yn * xn[i]
            i += 1

    def __mysign(self, x):
        if x < 0.0:
            return -1
        elif x > 0.0:
            return 1
        else:
            return 0

    def __h(self, w, xn):
        """
        Obtains raw w \dot x value unabridged
        w: the weight vector
        xn: the Xn vector
        """
        sum = 0.0
        xl = len(xn)
        i = 0
        while (i < xl):
            sum += w[i] * xn[i]
            i += 1
        return sum

    def __hsign(self, w, xn):
        """
        Obtains the classification of this point.
        w: the weight vector
        xn: the Xn vector
        returns: 0, 1, or -1. 0 means Not yet classified
        """
        sum = self.__h(w, xn)
        return self.__mysign(sum)

    def train(self, X, y, maxIterations=100000):
        lx = []
        for x in X:
            tmp = [1]
            tmp.extend(x)
            lx.append(tmp)

        plaW = [0] * len(lx[0])
        numIterations = 0

        while True: 
            misclassified = []
            for xn, yn in zip(lx, y):
                if (self.__hsign(plaW, xn) != yn):
                    misclassified.append((xn, yn))

            if len(misclassified) == 0:
                self._Ein = misclassified
                self._plaW = plaW
                self._numIterations = numIterations

                # complete, no more misclassified points
                return self

            ds = misclassified[random.randint(0, len(misclassified)-1)]
            self.__updatePLAWeights(plaW, ds[1], ds[0])

            numIterations += 1

            if numIterations >= maxIterations: 
                self._Ein = misclassified
                self._plaW = plaW
                self._numIterations = numIterations

                return self
    
    def test(self, X, y):
        plaW = self._plaW
        m, c = -plaW[1]/plaW[2], -plaW[0]/plaW[2]

        numDisagreement = 0
        for xn,yn in zip(X, y):
            if (yn != self.__mysign(xn[1] - m*xn[0] - c)):
                numDisagreement += 1

        return numDisagreement / len(y)
