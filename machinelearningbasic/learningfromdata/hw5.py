import os
import re
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from commons import generateRandomDataset

def Q1():
    s = 0.1
    d = 8
    for E in [0.007, 0.008, 0.009]:
        N = s**2 * (d+1) / (s**2 - E)
        print("Q1: E: {}, N: {}".format(E, N))

def Q4_5_6():
    # E(u,v) = (ue^v - 2ve^-u)^2
    # dE/du = 2 * (ue^v - 2ve^-u) * (e^v + 2ve^-u)
    # dE/dv = 2 * (ue^v - 2ve^-u) * (ue^v - 2e^-u)
    
    print("Q4: 2 * (ue^v - 2ve^-u) * (e^v + 2ve^-u)")

    u,v = (1,1)
    lr = 0.1
    I = 0
    while True:
        du = 2 * (u*math.exp(v) - 2*v*math.exp(-u)) * (math.exp(v) + 2*v*math.exp(-u))
        dv = 2 * (u*math.exp(v) - 2*v*math.exp(-u)) * (u*math.exp(v) - 2*math.exp(-u))
        u,v = (u - lr*du, v - lr*dv)
        Euv = (u*math.exp(v) - 2*v*math.exp(-u))**2
        if Euv < 1e-14:
            break
        I += 1
    
    print("Q5: I = {}\nQ6: (u,v)=({}, {})".format(I, u, v))

def Q7():
    u,v = (1,1)
    lr = 0.1
    I = 0
    totalIterations = 1000
    while I < totalIterations:
        du = 2 * (u*math.exp(v) - 2*v*math.exp(-u)) * (math.exp(v) + 2*v*math.exp(-u))
        u,v = (u - lr*du, v)

        dv = 2 * (u*math.exp(v) - 2*v*math.exp(-u)) * (u*math.exp(v) - 2*math.exp(-u))
        u,v = (u, v - lr*dv)
        I += 1
    
    Euv = (u*math.exp(v) - 2*v*math.exp(-u))**2
    print("Q7: ({},{}) => E(u,v):{}".format(u, v, Euv))

def Q8_9(N, totalIterations):
    _I = 0
    sumEout = 0.0
    while _I < totalIterations:
        X, y, m, c = generateRandomDataset(N)
        logreg = LogisticRegression()
        logreg.train(X, y, True, 0.01, 0.01)
        Xtest, ytest, m, c = generateRandomDataset(N, m, c)
        sumEout += logreg.test(Xtest, ytest)
        _I += 1
    print("Q8. avg Eout: {}".format(sumEout / totalIterations))
    print("Q9. epochs: {}".format(logreg._epochs))

def main():
    Q8_9(100, 100)

if __name__ == '__main__':
    main()
