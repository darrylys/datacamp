import os
import re
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import commons

from sklearn import svm
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

def Q7():
    df_train, df_test = commons.featuresDatasetLoadTrainTest()
    X_labels = ['intensity', 'symmetry']
    y_label = 'label'
    
    minDigit = 0
    minEin = math.inf

    for digit in [5,6,7,8,9]:
        df5_train = commons.featuresDatasetOneVsAllCleanup(df_train, digit)
        df5_test = commons.featuresDatasetOneVsAllCleanup(df_test, digit)

        X_train = df5_train[X_labels]
        y_train = df5_train[y_label]

        X_test = df5_test[X_labels]
        y_test = df5_test[y_label]

        ridge = linear_model.Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)

        y_train_predicted = [commons.hsign(x) for x in ridge.predict(X_train)]
        Ein = 1 - accuracy_score(y_train, y_train_predicted)

        if minEin > Ein:
            minEin = Ein
            minDigit = digit

    print("Q7: lowest Ein classifier: {} vs all".format(minDigit))

def Q8():
    df_train, df_test = commons.featuresDatasetLoadTrainTest()
    X_labels = ['intensity', 'symmetry']
    y_label = 'label'
    transformed_columns = ['x1', 'x2', 'x1x2', 'x1^2', 'x2^2']

    minDigit = 0
    minEout = math.inf

    for digit in [0,1,2,3,4]:
        dfd_train = commons.featuresDatasetOneVsAllCleanup(df_train, digit)
        dfd_test = commons.featuresDatasetOneVsAllCleanup(df_test, digit)

        X_train = [[x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2] for x in dfd_train[X_labels].to_numpy()]
        X_train = pd.DataFrame.from_records(X_train, columns=transformed_columns)
        y_train = dfd_train[y_label]

        ridge = linear_model.Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)

        X_test = [[x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2] for x in dfd_test[X_labels].to_numpy()]
        X_test = pd.DataFrame.from_records(X_test, columns=transformed_columns)
        y_test = dfd_test[y_label]

        y_test_predicted = [commons.hsign(x) for x in ridge.predict(X_test)]
        Eout = 1 - accuracy_score(y_test, y_test_predicted)

        if minEout > Eout:
            minEout = Eout
            minDigit = digit

    print("Q8: lowest Eout classifier: {} vs all".format(minDigit))

def train_test_plain(df_train, df_test, X_labels, y_label):
    X_train = df_train[X_labels]
    y_train = df_train[y_label]

    X_test = df_test[X_labels]
    y_test = df_test[y_label]

    ridge = linear_model.Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    Ein = 1 - accuracy_score(y_train, [commons.hsign(x) for x in ridge.predict(X_train)])
    Eout = 1 - accuracy_score(y_test, [commons.hsign(x) for x in ridge.predict(X_test)])

    return Ein, Eout

def train_test_transformed(df_train, df_test, X_labels, y_label, transformed_columns, fnTransform, vlambda=1.0):
    X_train = [fnTransform(x) for x in df_train[X_labels].to_numpy()]
    X_train = pd.DataFrame.from_records(X_train, columns=transformed_columns)
    y_train = df_train[y_label]

    X_test = [fnTransform(x) for x in df_test[X_labels].to_numpy()]
    X_test = pd.DataFrame.from_records(X_test, columns=transformed_columns)
    y_test = df_test[y_label]

    ridge = linear_model.Ridge(alpha=vlambda)
    ridge.fit(X_train, y_train)

    Ein = 1 - accuracy_score(y_train, [commons.hsign(x) for x in ridge.predict(X_train)])
    Eout = 1 - accuracy_score(y_test, [commons.hsign(x) for x in ridge.predict(X_test)])

    return Ein, Eout

def transform(x):
    return [x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2]

def Q9():
    df_train, df_test = commons.featuresDatasetLoadTrainTest()
    X_labels = ['intensity', 'symmetry']
    y_label = 'label'
    transformed_columns = ['x1', 'x2', 'x1x2', 'x1^2', 'x2^2']
    digits = [0,1,2,3,4,5,6,7,8,9]

    pein, peout, tein, teout = [], [], [], []

    for digit in digits:
        dfd_train = commons.featuresDatasetOneVsAllCleanup(df_train, digit)
        dfd_test = commons.featuresDatasetOneVsAllCleanup(df_test, digit)

        plainEin, plainEout = train_test_plain(dfd_train, dfd_test, X_labels, y_label)
        trfEin, trfEout = train_test_transformed(dfd_train, dfd_test, X_labels, y_label, transformed_columns, transform)

        pein.append(plainEin)
        peout.append(plainEout)
        tein.append(trfEin)
        teout.append(trfEout)

    ans = [0]*5

    # [a] Overtting always occurs when we use the transform.
    ans[0] = 1
    for digit in digits:
        i = digit
        if not (tein[i] < pein[i] and teout[i] > peout[i]):
            ans[0] = -1

    # [b] (Eout with transform <= 0.95 * Eout without transform).
    ans[1] = 1
    for digit in digits:
        i = digit
        if not (teout[i] <= 0.95 * peout[i]):
            ans[1] = -1

    # [c] The transform does not make any difference in the out-of-sample performance.
    # ASSUMING that the performance difference is < 0.0001
    # not 0.05 is used because ALL Eout differences are below that, and option [e] is correct anyway
    ans[2] = 1
    for digit in digits:
        i = digit
        if abs(teout[i] - peout[i]) >= 0.0001:
            ans[2] = -1

    # [d] (Eout without transform <= 0.95 * Eout with transform).
    ans[3] = 1
    for digit in digits:
        i = digit
        if not (peout[i] <= 0.95 * teout[i]):
            ans[3] = -1

    # [e] The transform improves the out-of-sample performance of `5 versus all,' but by less than 5%.
    if teout[5] / peout[5] > 0.95:
        ans[4] = 1
    else:
        ans[4] = -1
    
    print("pein: {}\npeout: {}\ntein: {}\nteout: {}".format(pein, peout, tein, teout))
    print("Q9. 1 = right, -1 = wrong. ans: [a, b, c, d, e]: {}".format(ans))

def Q10():
    df_train, df_test = commons.featuresDatasetLoadTrainTest()
    X_labels = ['intensity', 'symmetry']
    y_label = 'label'
    transformed_columns = ['x1', 'x2', 'x1x2', 'x1^2', 'x2^2']

    df15_train = commons.featuresDatasetOneVsOneCleanup(df_train, 1, 5)
    df15_test = commons.featuresDatasetOneVsOneCleanup(df_test, 1, 5)

    lein, leout = [], []
    for vlambda in [1, 0.01]:
        Ein, Eout = train_test_transformed(df15_train, df15_test, X_labels, y_label, transformed_columns, transform, vlambda)
        
        lein.append(Ein)
        leout.append(Eout)

    ans = [0]*5

    # [a] Overtting occurs (from lmd = 1 to lmd = 0:01).
    if lein[1] < lein[0] and leout[1] > leout[0]:
        ans[0] = 1
    else:
        ans[0] = -1
    
    # [b] The two classiers have the same Ein.
    if lein[0] == lein[1]:
        ans[1] = 1
    else:
        ans[1] = -1

    # [c] The two classiers have the same Eout.
    if leout[0] == leout[1]:
        ans[2] = 1
    else:
        ans[2] = -1

    # [d] When lmd goes up, both Ein and Eout go up.
    if lein[1] > lein[0] and leout[1] > leout[0]:
        ans[3] = 1
    else:
        ans[3] = -1

    # [e] When lmd goes up, both Ein and Eout go down.
    if lein[1] < lein[0] and leout[1] < leout[0]:
        ans[4] = 1
    else:
        ans[4] = -1

    print("Q10. 1 = right, -1 = wrong. ans: [a, b, d, c, e]: {}".format(ans))

def Q11():
    X = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
    y = [-1, -1, -1, 1, 1, 1, 1]

    Xt = [[x[1]**2 - 2*x[0] - 1, x[0]**2 - 2*x[1] + 1] for x in X]
    plt.scatter([xt[0] for xt in Xt[0:3]], [xt[1] for xt in Xt[0:3]], c='red')
    plt.scatter([xt[0] for xt in Xt[3:]], [xt[1] for xt in Xt[3:]], c='blue')
    plt.show()

def Q12():
    X = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
    y = [-1, -1, -1, 1, 1, 1, 1]

    svc = svm.SVC(C=math.inf, kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    svc.fit(X, y)
    print("Q12: {}".format(np.sum(svc.n_support_)))

def main():
    Q12()


if __name__ == '__main__':
    main()
