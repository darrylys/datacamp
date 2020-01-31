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

def loadDs():
    df_train = pd.read_csv('features.train', sep=r"\s+", names=['digit', 'intensity', 'symmetry'], header=None)
    df_test = pd.read_csv('features.test', sep=r"\s+", names=['digit', 'intensity', 'symmetry'], header=None)
    return df_train, df_test

def Q2():
    df_train, df_test = loadDs()

    maxEin = -1
    maxDigit = 0
    maxNSupportVector = 0

    for digit in [0, 2, 4, 6, 8]:
        df_train.loc[df_train['digit'] == digit, 'label'] = 1
        df_train.loc[df_train['digit'] != digit, 'label'] = -1

        X_train = df_train[['intensity', 'symmetry']]
        y_train = df_train['label']

        svc = svm.SVC(C = 0.01, kernel='poly', degree=2, coef0=1.0, gamma=1.0)
        svc.fit(X_train, y_train)
        
        y_train_hat = svc.predict(X_train)
        ein = 1 - accuracy_score(y_train, y_train_hat)

        if ein > maxEin:
            maxEin = ein
            maxDigit = digit
            maxNSupportVector = np.sum(svc.n_support_)

    print("Q2: maxEin: {}, maxDigit: {}".format(maxEin, maxDigit))
    return maxNSupportVector
    
def Q3():
    df_train, df_test = loadDs()

    minEin = math.inf
    minDigit = 0
    minNSupportVector = 0

    for digit in [1, 3, 5, 7, 9]:
        df_train.loc[df_train['digit'] == digit, 'label'] = 1
        df_train.loc[df_train['digit'] != digit, 'label'] = -1

        X_train = df_train[['intensity', 'symmetry']]
        y_train = df_train['label']

        svc = svm.SVC(C = 0.01, kernel='poly', degree=2, coef0=1.0, gamma=1.0)
        svc.fit(X_train, y_train)
        
        y_train_hat = svc.predict(X_train)
        ein = 1 - accuracy_score(y_train, y_train_hat)

        if ein < minEin:
            minEin = ein
            minDigit = digit
            minNSupportVector = np.sum(svc.n_support_)

    print("Q3: minEin: {}, minDigit: {}".format(minEin, minDigit))
    return minNSupportVector

def OneVsOneCleanup(df, digit1, digit2):
    df.loc[:, 'label'] = 0
    df.loc[df['digit'] == digit1, 'label'] = 1
    df.loc[df['digit'] == digit2, 'label'] = -1

    cleanDf = df[df.label != 0]
    return cleanDf

def trainSVM_5_6(degree, clist = [0.001, 0.01, 0.1, 1]):
    df_train, df_test = loadDs()

    spv = []
    eol = []
    eil = []

    for C in clist:
        cleanTrainDf = OneVsOneCleanup(df_train, 1, 5)
        cleanTestDf = OneVsOneCleanup(df_test, 1, 5)

        X_train = cleanTrainDf[['intensity', 'symmetry']]
        y_train = cleanTrainDf['label']

        X_test = cleanTestDf[['intensity', 'symmetry']]
        y_test = cleanTestDf['label']

        svc = svm.SVC(C = C, kernel = 'poly', degree = degree, coef0 = 1.0, gamma=1.0)
        svc.fit(X_train, y_train)

        y_train_predicted = svc.predict(X_train)
        y_test_predicted = svc.predict(X_test)

        nSupportVec = np.sum(svc.n_support_)
        Eout = 1 - accuracy_score(y_test, y_test_predicted)
        Ein = 1 - accuracy_score(y_train, y_train_predicted)

        spv.append(nSupportVec)
        eol.append(Eout)
        eil.append(Ein)
    
    return spv, eol, eil

def Q6():
    spv2, eol2, eil2 = trainSVM_5_6(2, [0.0001, 0.001, 0.01, 0.1, 1])
    spv5, eol5, eil5 = trainSVM_5_6(5, [0.0001, 0.001, 0.01, 0.1, 1])

    ans = []

    if eil5[0] > eil2[0]:
        ans.append("[a]")
    
    if spv5[1] < spv2[1]:
        ans.append("[b]")
    
    if eil5[2] > eil2[2]:
        ans.append("[c]")

    if eol5[4] < eol2[4]:
        ans.append("[d]")
    
    if len(ans) == 0:
        ans.append("[e]")

    print("Q6: ans: {}".format(ans))

def Q5():
    spv, eol, eil = trainSVM_5_6(2)

    increasing = 0
    decreasing = 0
    i = 1
    while i < len(spv):
        if spv[i-1] < spv[i]:
            increasing += 1
        elif spv[i-1] > spv[i]:
            decreasing += 1
        i += 1

    ans = []
    if increasing == len(spv)-1:
        ans.append("[b]")
    elif decreasing == len(spv)-1:
        ans.append("[a]")
    
    decreasing = 0
    i = 1
    while i < len(eol):
        if eol[i-1] > eol[i]:
            decreasing += 1
        i += 1
    if decreasing == len(eol)-1:
        ans.append("[c]")

    meil = min(eil)
    if meil == eil[3]:
        ans.append("[d]")

    if len(ans) == 0:
        ans.append("[e]")

    print("spv: {}, eol: {}, eil: {}".format(spv, eol, eil))
    print("Q5: {}".format(ans))

def main():
    #a = Q2()
    #b = Q3()
    #print("Q4: diff: {}".format(abs(a-b)))
    #Q5()
    Q6()
    pass

if __name__ == '__main__':
    main()
