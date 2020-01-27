import os
import re
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_regression import LinearRegression

def readInOutDta():
    df_train = pd.read_csv('in.dta', sep=r"\s+", names=['x1', 'x2', 'label'], header=None)
    df_test = pd.read_csv('out.dta', sep=r"\s+", names=['x1', 'x2', 'label'], header=None)
    X_train = [[x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], abs(x[0]-x[1]), abs(x[0]+x[1])] for x in df_train[['x1', 'x2']].to_numpy()]
    y_train = df_train['label'].to_numpy()
    X_test = [[x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], abs(x[0]-x[1]), abs(x[0]+x[1])] for x in df_test[['x1', 'x2']].to_numpy()]
    y_test = df_test['label'].to_numpy()
    return X_train, X_test, y_train, y_test

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

