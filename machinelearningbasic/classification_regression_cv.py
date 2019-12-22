# src: 
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
# https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833
# https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib
#
# purpose: handling over/under fitting of data
# Overfitting: The algorithm is really accurate in train data, but
# cannot be generalized due to picking up noise signals in train data.
# Underfitting: The algorithm does not pick up trends in train data,
# and thus, cannot be generalized.
# CV is to handle overfitting, than underfitting.

import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import pandas as pd

def createImg(X, y, X_test, y_pred, figTitle="HDI", xlabel="Year", ylabel="HDI"):
    fig = plt.figure(figsize=(4, 5))
    plt.scatter(X, y,  color='black') 
    plt.title(figTitle) 
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 
    plt.plot(X_test, y_pred, color='red',linewidth=2) 
    plt.savefig("{}.png".format(figTitle))
    plt.close(fig)

def manualCV(X, y):
    """
    Manually calculate score using Cross Validation 5-fold
    X : numpy array of features
    y : numpy array of result
    """

    kf = KFold(n_splits=5)
    lgr = linear_model.LinearRegression()
    scores = []

    I = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgr.fit(X_train, y_train)

        # R2 score range is in (-inf, 1.0]
        # best score is 1.
        scores.append(model.score(X_test, y_test))
        createImg(X, y, X, model.predict(X), "HDI-{}".format(I))

        I += 1
    
    print(scores)
    print(np.mean(scores))

def fastCV(X, y):
    """
    Does the same thing as manualCV, but using cross_val_score
    convenience function
    X: numpy array of features
    y: numpy array of result
    """
    
    lgr = linear_model.LinearRegression()
    scores = cross_val_score(lgr, X, y, cv=5)

    print(scores)
    print(np.mean(scores))

df = pd.read_csv("hdi_human_development_index.csv")

df.set_index('country', inplace=True)

country = 'Brazil'
bdf = df.loc[country]

X = np.asarray([[int(x)] for x in df.columns])
y = bdf.values

print("Manual CV calculating score")
manualCV(X, y)

print("Using cross_val_score")
fastCV(X, y)
