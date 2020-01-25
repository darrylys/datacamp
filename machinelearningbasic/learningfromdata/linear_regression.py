
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

class LinearRegression:
    def __init__(self):
        super().__init__()
    
    def __mysign(self, x):
        if x < 0.0:
            return -1
        elif x > 0.0:
            return 1
        else:
            return 0

    def __h(self, w, xn):
        """
        Obtains raw w dot x value unabridged
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

    def oz(self, a, b):
        if a == b:
            return 0
        else:
            return 1

    def train(self, X, y, addYIntercept=True):
        w0x = []
        self._addYIntercept = addYIntercept
        for x in X:
            if addYIntercept:
                w0x.append([1] + x)
            else:
                w0x.append([0] + x)

        aX = np.array(w0x)
        ay = np.array(y)
        
        piaX = np.linalg.pinv(aX)
        w = np.dot(piaX, ay)
        
        Ein = 0
        Ein1 = 0

        N = len(y)
        for xn,yn in zip(w0x, y):
            Ein += (self.__hsign(w, xn) - yn)**2
            Ein1 += self.oz(self.__hsign(w, xn), yn)

        self._ein = Ein/N
        self._ein1 = Ein1/N
        self._w = w

        return self

    def regressionTest(self, X, y):
        sumDisagreement = 0
        w = self._w
        for xn,yn in zip(X, y):
            if self._addYIntercept:
                mxn = [1] + xn
            else:
                mxn = [0] + xn
            sumDisagreement += (yn - self.__h(w, mxn))**2
        return sumDisagreement / len(y)

    def test(self, X, y):
        numDisagreement = 0
        w = self._w
        for xn,yn in zip(X, y):
            if self._addYIntercept:
                mxn = [1] + xn
            else:
                mxn = [0] + xn
            if yn != self.__hsign(w, mxn):
                numDisagreement += 1
        return numDisagreement / len(y)


def main():
    a = np.array([[1, 2], [1, 3], [1, 4]])
    print(a)

    b = np.linalg.pinv(a)
    print(b)

    ab = np.dot(a, b)
    print(ab)

    v = np.array([4, 5, 6])
    bv = np.dot(b, v)
    print(bv)

    print(v[2])


if __name__ == '__main__':
    main()
