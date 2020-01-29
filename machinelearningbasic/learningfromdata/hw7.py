import os
import re
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from linear_regression import LinearRegression
import commons

def validationTest(X_train, y_train, X_val, y_val, X_test_all, y_test_all):
    minEval = math.inf
    minEvalK = 0

    minEout = math.inf
    minEoutK = 0

    #print("X_train: {}".format(X_train))
    #print("y_train: {}".format(y_train))
    #print("X_val: {}".format(X_val))
    #print("y_val: {}".format(y_val))
    #print("X_test_all: {}".format(X_test_all))
    #print("y_test_all: {}".format(y_test_all))

    for k in [3,4,5,6,7]:
        linreg = LinearRegression()
        linreg.train(X_train.iloc[:,0:k].to_numpy(), y_train.to_numpy())
        Eval = linreg.test(X_val.iloc[:,0:k].to_numpy(), y_val.to_numpy())
        if Eval < minEval:
            minEval = Eval
            minEvalK = k

        Eout = linreg.test(X_test_all.iloc[:,0:k].to_numpy(), y_test_all.to_numpy())
        if Eout < minEout:
            minEout = Eout
            minEoutK = k

    return minEval, minEvalK, minEout, minEoutK

def Q1_2_5():
    X_train_all, X_test_all, y_train_all, y_test_all = commons.readInOutDtaPandas()
    X_train = X_train_all.iloc[0:25]
    y_train = y_train_all.iloc[0:25]
    X_val = X_train_all.iloc[25:]
    y_val = y_train_all.iloc[25:]

    minEval, minEvalK, minEout, minEoutK = validationTest(X_train, y_train, X_val, 
    y_val, X_test_all, y_test_all)

    print("Q1: minEval: {}, minEvalK: {}".format(minEval, minEvalK))
    print("Q2: minEout: {}, Q5: minEoutK: {}".format(minEout, minEoutK))

def Q3_4_5():
    X_train_all, X_test_all, y_train_all, y_test_all = commons.readInOutDtaPandas()
    X_train = X_train_all.iloc[25:]
    y_train = y_train_all.iloc[25:]
    X_val = X_train_all.iloc[0:25]
    y_val = y_train_all.iloc[0:25]

    minEval, minEvalK, minEout, minEoutK = validationTest(X_train, y_train, X_val, 
    y_val, X_test_all, y_test_all)

    print("Q3: minEval: {}, minEvalK: {}".format(minEval, minEvalK))
    print("Q4: minEout: {}, Q5: minEoutK: {}".format(minEout, minEoutK))

def Q6():
    iterations = 100000
    _I = 0

    sume1 = 0
    sume2 = 0
    sume = 0
    while _I < iterations:
        e1 = random.uniform(0, 1)
        e2 = random.uniform(0, 1)
        e = min(e1, e2)

        sume1 += e1
        sume2 += e2
        sume += e
        _I += 1

    print("Q6: E1: {}, E2: {}, E: {}".format(sume1 / iterations, sume2 / iterations, sume / iterations))

def main():
    #Q1_2_5()
    #Q3_4_5()
    Q6()

if __name__ == '__main__':
    main()
