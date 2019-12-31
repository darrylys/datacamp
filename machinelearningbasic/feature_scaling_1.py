# Feature scaling
# src: 
# https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
# https://www.datacamp.com/community/tutorials/preprocessing-in-data-science-part-1-centering-scaling-and-knn
# https://medium.com/@swethalakshmanan14/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff
# 
# Feature scaling and centering is not useful for tree-based classifiers.
# Normalization:
# > scaling a dataset so that its minimum is 0 and its maximum 1 (e.g. Min/Max normalization)
# > good when the data has unknown distribution, or the distribution is not Gaussian
# > works well with ML that does not assume the data distribution such as KNN and ANN.
# > Results in smaller standard deviation because the max and min is bounded.
# > More sensitive to outliers compared to Standardization
#
# Standardization (Z-score normalization):
# > it's job is to center the data around 0 and to scale with respect to the standard deviation.
#       x' = (x - \avg) / \stdev
# > good when the data has gaussian distribution
# > works well with ML that assumes gaussian distribution (LogReg, LinReg, Linear Discriminant Analysis)
#
# centering a variable: 
# subtracting the mean of the variable from each 
# data point so that the new variable's mean is 0; 
# 
# scaling a variable: 
# multiplying each data point by a constant in 
# order to alter the range of the data
#
# wine_dataset.csv src: 
# https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv
# 
# winequality dataset src:
# http://archive.ics.uci.edu/ml/datasets/Wine+Quality
#

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

def evaluate(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print("Precision: {}".format(precision_score(y_true, y_pred)))
    print("Recall: {}".format(recall_score(y_true, y_pred)))
    print("F1-Score: {}".format(f1_score(y_true, y_pred)))
    print("AUC: {}".format(roc_auc_score(y_true, y_pred)))

def trainmodelcv(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    y_pred = cross_val_predict(model, X, y, cv=5)
    print(scores)
    print(np.mean(scores))
    evaluate(y, y_pred)

def winequality():
    train = pd.read_csv('winequality-red.csv', delimiter=';')

    featureCols = ["fixed acidity","volatile acidity","citric acid",
            "residual sugar","chlorides","free sulfur dioxide",
            "total sulfur dioxide","density","pH","sulphates",
            "alcohol"]
    classCol = "quality"
    classColBinary = 'isWineGood'

    train.loc[train[classCol] > 5, classColBinary] = 1
    train.loc[train[classCol] <= 5, classColBinary] = 0

    if True:
        # accuracy: 0.5922
        print("\nPlain un-preprocessed KNN scores: ")
        trainmodelcv(KNeighborsClassifier(), train[featureCols], train[classColBinary])

    if True:
        # accuracy: 0.6754
        print("\nStd-scaled preprocessed KNN scores: ")
        std_scaler = preprocessing.StandardScaler().fit(train[featureCols])
        train_scaled = std_scaler.transform(train[featureCols])
        trainmodelcv(KNeighborsClassifier(), train_scaled, train[classColBinary])

    if True:
        # accuracy: 0.6729, not very different to std-scaled
        print("\nMinMax-scaled preprocessed KNN scores: ")
        std_scaler = preprocessing.MinMaxScaler().fit(train[featureCols])
        train_scaled = std_scaler.transform(train[featureCols])
        trainmodelcv(KNeighborsClassifier(), train_scaled, train[classColBinary])
    



def main():
    winequality()

if __name__ == '__main__':
    main()
