import os
import re
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_regression import LinearRegression
import commons

def readInOutDta():
    return commons.readInOutDta()

def wdkret(weightDecayLambdaValue):
    X_train, X_test, y_train, y_test = readInOutDta()

    linreg = LinearRegression()
    linreg.train(X_train, y_train, weightDecayLambda=weightDecayLambdaValue)
    Ein = linreg._ein1
    Eout = linreg.test(X_test, y_test)

    return linreg, Ein, Eout

def wdk(qn, weightDecayLambdaValue):
    _, Ein, Eout = wdkret(weightDecayLambdaValue)

    print("Q{}. Ein: {}, Eout: {}".format(qn, Ein, Eout))

def Q5_6():
    kl = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
    smallestK = 0
    minv = math.inf
    for k in kl:
        _, _, Eout = wdkret(k)
        print("k: {} Eout: {}".format(k, Eout))
        if Eout < minv:
            minv = Eout
            smallestK = k
    
    print("Q5: smallest K: {}, min Eout: {}".format(smallestK, minv))

def Q4():
    wdk(4, 10**3)

def Q3():
    wdk(3, 10**-3)

def Q2():
    wdk(2, 0)

def main():
    pass
    #Q2()
    #Q3()
    #Q4()
    #Q5_6()

if __name__ == '__main__':
    main()

