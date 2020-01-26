
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

class LogisticRegression:
    def __init__(self):
        super().__init__()

    def __crossEntropy(self, W, X, y):
        sum = 0.0
        for Xn,yn in zip(X, y):
            sum += math.log(1 + math.exp(-yn * np.dot(W, Xn)))
        return sum / len(y)

    def __stochasticDelEin(self, W, X, y, idxr=None):
        N = len(y)
        if not idxr:
            idxr = random.randint(0, N-1)

        Xn = X[idxr]
        yn = y[idxr]
        mul = yn / (1 + math.exp(yn * np.dot(W, Xn)))
        return np.array([mul * xn for xn in Xn])

    def train(self, X, y, addYIntercept = True, tolerance=0.01, learning_rate=0.01):
        self._addYIntercept = addYIntercept
        
        Xtrain = np.array([[1] + x if addYIntercept else [0] + x for x in X])
        ytrain = np.array(y)

        baseIdx = range(0, len(ytrain))
        t = 0
        epochs = 0
        w = np.array([0] * len(Xtrain[0]))
        executionRunning = True
        while executionRunning:
            idxPermuted = np.random.permutation(baseIdx)
            wep = w
            for idx in idxPermuted:
                delEin = -1 * self.__stochasticDelEin(w, Xtrain, ytrain, idx)
                delw = np.array([learning_rate * ein for ein in delEin])
                wt = np.subtract(w, delw)
                w = wt
                t += 1

            dw = np.subtract(wep, w)
            normDw = np.linalg.norm(dw) # default is norm-2 or Euclidean norm
            if (normDw < tolerance):
                executionRunning = False
                break
            
            #print("epochs: {}, w: {}".format(epochs, w))
            epochs += 1
        
        self._epochs = epochs
        self._w = wt
        self._t = t
        return self

    def test(self, X, y):
        Xtest = np.array([[1] + x if self._addYIntercept else [0] + x for x in X])
        ytest = y
        return self.__crossEntropy(self._w, Xtest, ytest)


