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
from perceptron import Perceptron

from sklearn import svm
from sklearn.metrics import accuracy_score

def validationTest(X_train, y_train, X_val, y_val, X_test_all, y_test_all):
    minEval = math.inf
    minEvalK = 0

    minEout = math.inf
    minEoutK = 0

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

def countValues(Y):
    a = {}
    for y in Y:
        if y in a:
            a[y] += 1
        else:
            a[y] = 1
    return a

def PLAvsSVM(N, totalIterations, Ntests):
    _I = 0
    _clfIsBetter = 0
    _clfSupportN = 0

    while _I < totalIterations:
        X, y, m, c = [], [], 0, 0
        while True:
            X, y, m, c = commons.generateRandomDataset(N)
            counts = countValues(y)
            if len(counts) >= 2:
                break
        
        svc = svm.SVC(kernel='linear', C=math.inf)
        clf = svc.fit(X, y)
        
        pla = Perceptron()
        pla.train(X, y)

        Xtest, ytest, m, c = commons.generateRandomDataset(Ntests, m, c)

        plaErr = pla.test(Xtest, ytest)
        
        clfYhat = clf.predict(Xtest)
        clfErr = 1 - accuracy_score(ytest, clfYhat)

        if (plaErr > clfErr):
            _clfIsBetter += 1

        _clfSupportN += np.sum(clf.n_support_)

        _I += 1
    
    return (_clfIsBetter / totalIterations), (_clfSupportN / totalIterations)

def Q8():
    cp, cn = PLAvsSVM(10, 1000, 1000)
    print("Q8: % SVM better: {}, avg Support: {}".format(cp, cn))

def Q9_10():
    cp, cn = PLAvsSVM(100, 1000, 1000)
    print("Q9: % SVM better: {}, Q10: avg Support: {}".format(cp, cn))

def main():
    #Q1_2_5()
    #Q3_4_5()
    #Q6()
    #Q8()
    #Q9_10()
    #PLAvsSVM(10, 1, 1000)
    pass

if __name__ == '__main__':
    main()
