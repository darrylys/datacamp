
# Highest score: 0.79904, rank: ~1700
#

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import seaborn as sns

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import xgboost

def sandbox():
    x = [i**2 for i in range(0, 20)]
    
    # bins = number of buckets 
    # maximum x value is 400, so, if bins = 40, means, 
    # the histogram is divided by 40 boxes, each of size 10.
    # [0, 10), [10, 20), ..., [390, 400)
    # the x-es will be put into these boxes.
    sns.distplot(x, kde=False, bins=40)
    plt.show()

def displayAge(data, title, subplot):
    plt.subplot(subplot)
    plt.tight_layout()
    sns.distplot(data.Age.dropna(), bins=20, kde=False)
    plt.title(title)
    plt.legend()

def displayAgeVsSurvivedPlot(data, title, subplot):
    plt.subplot(subplot)
    plt.tight_layout()
    sns.distplot(data[data["Survived"] == 1].Age.dropna(), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Age.dropna(), bins=20, kde=False, label="dead")
    plt.title(title)
    plt.legend()

def displayFareVsSurvivedPlot(data, title, subplot):
    plt.subplot(subplot)
    plt.tight_layout()
    sns.distplot(data[data["Survived"] == 1].Fare.dropna(), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Fare.dropna(), bins=20, kde=False, label="dead")
    plt.title(title)
    plt.legend()

def displayXNPlot(data, title, subplot, X, N):
    plt.subplot(subplot)
    plt.tight_layout()
    sns.countplot(x = X, data=data.dropna(), hue=N)
    plt.title(title)
    plt.legend()

def putToBins(data, bins, colName):
    return pd.cut(data[colName], bins=bins, 
            labels=[x for x in range(1, len(bins))]).astype(int)

def fillMissingAge(row):
    smarterMedianAge = {
        #(sex, pclass, title)
        ('female', 1, 2) : 29.85,
        ('female', 1, 3) : 38.50,
        ('female', 1, 5) : 49.00,
        ('female', 1, 7) : 33.00,
        ('female', 2, 2) : 24.00,
        ('female', 2, 3) : 31.50,
        ('female', 3, 2) : 22.00,
        ('female', 3, 3) : 29.70,
        ('male', 1, 1) : 36.00,
        ('male', 1, 4) :  4.00,
        ('male', 1, 5) : 40.00,
        ('male', 1, 6) : 56.00,
        ('male', 2, 1) : 30.00,
        ('male', 2, 4) :  1.00,
        ('male', 2, 5) : 46.50,
        ('male', 3, 1) : 29.70,
        ('male', 3, 4) :  6.50
    }

    if np.isnan(row["Age"]):
        key = (row['Sex'], row['Pclass'], row['Title'])
        return smarterMedianAge[key]
    
    else:
        return row["Age"]


def transformData(data):
    data["SexI"] = data["Sex"].map({"male": 1, "female": 2})
    #data["IsMale"] = data["Sex"].map({"male": 1, "female": 0})
    
    data["Embarked"] = data["Embarked"].fillna("S")
    data["EmbarkedI"] = data["Embarked"].map({"S": 1, "C": 2, "Q": 3})
    #data["EmbarkedS"] = data["Embarked"].map({"S": 1, "C": 0, "Q": 0})
    #data["EmbarkedC"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 0})
    #data["EmbarkedQ"] = data["Embarked"].map({"S": 0, "C": 0, "Q": 1})

    #data["Pclass3"] = data["Pclass"].map({3 : 1, 2 : 0, 1 : 0})
    #data["Pclass2"] = data["Pclass"].map({3 : 0, 2 : 1, 1 : 0})
    #data["Pclass1"] = data["Pclass"].map({3 : 0, 2 : 0, 1 : 1})

    # title is good predictor, but not improv AUC/Accuracy. Kaggle score: 0.75
    data["Title"] = data["Name"].map(lambda x : getTitleFromName(x)).astype(int)
    #print(data["Title"].value_counts())

    #SibSp,Parch, Fare
    # age is categorized to specific bins
    ageBins = [0, 18, 23, 28, 34, 44, 200]
    data["Age"] = data.apply(fillMissingAge, axis=1)
    #data["Age"] = data["Age"].fillna(29.7) # median of age
    data["AgeBin"] = pd.cut(data["Age"], bins=ageBins, labels=[x for x in range(1, len(ageBins))]).astype(int)

    fareBins = [-1, 7.75, 8.05, 12.475, 19.258, 27.9, 56.929, 1000]
    # fare can be 0, NaN, huh.
    data["Fare"] = data["Fare"].fillna(8.05) # median fare of Embarked == S and Pclass == 3
    data["FareBin"] = pd.cut(data["Fare"], bins=fareBins, labels=[x for x in range(1, len(fareBins))]).astype(int)

    data["Relatives"] = data["SibSp"] + data["Parch"]
    data.loc[data['Relatives'] > 0, 'alone'] = 0
    data.loc[data['Relatives'] == 0, 'alone'] = 1
    data['alone'] = data['alone'].astype(int)

    return data

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

def predict(model, X, y, xtest):
    model.fit(X, y)
    return model.predict(xtest)

def getTitleFromName(strName):
    titleMap = {
        "mr": 1,
        "miss": 2,
        "mrs": 3,
        "master": 4,
        "rev": 5,
        "col": 6,
        "dona": 7,
        "dr": 5,
        "ms": 3,
        "mlle": 2,
        "major": 6,
        "don": 5,
        "mme": 7,
        "capt": 6,
        "jonkheer": 5,
        "sir": 5,
        "the countess": 7,
        "lady": 7
    }

    givenName = strName.split(",")[1]
    title = givenName.split(".")[0].lower().strip()
    return titleMap[title]


def trainmodel(model, data, test=None):
    print("Training model: {}", model)
    #featscols = ["SibSp","Parch","IsFemale",
    #"IsMale","EmbarkedS","EmbarkedC","EmbarkedQ","Pclass3",
    #"Pclass2","Pclass1","AgeBin","FareBin"]

    # score: 0.758
    #featscols = ["SibSp","Parch","SexI","EmbarkedI","Pclass","AgeBin","FareBin"]

    # score: 0.799 (with RF opti2)
    featscols = ["SibSp","Parch","SexI","EmbarkedI","Pclass","AgeBin","FareBin","Title"]

    # score: 0.779 (with RF opti2)
    #featscols = ["Relatives","SexI","EmbarkedI","Pclass","AgeBin","FareBin","Title"]

    # score: 0.751
    #featscols = ["alone","SexI","EmbarkedI","Pclass","AgeBin","FareBin"]
    #featscols = ["SibSp","Parch","IsFemale",
    #"IsMale","Pclass3","Pclass2","Pclass1","AgeBin","FareBin"]
    targetcol = "Survived"
    trainmodelcv(model, data[featscols], data[targetcol])

    model2 = RandomForestClassifier()
    model2.fit(data[featscols], data[targetcol])
    importances = pd.DataFrame({'feature':featscols,'importance':np.round(model2.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)

    if (test is not None):
        #print(test.info())
        #print(test[test["Fare"].isna()])

        testpred = predict(model, data[featscols], data[targetcol], test[featscols])
        test["Survived"] = testpred
        test[["PassengerId", "Survived"]].to_csv("test.predicted.csv", index=False)

    print("=============================================")

# kaggle score: 0.75
def trainRFBasicModel(data, test=None):
    rf = RandomForestClassifier(oob_score=True)
    trainmodel(rf, data, test)

# kaggle score: 0.77
def trainRFOptimized1Model(data, test=None):
    rf = RandomForestClassifier(n_estimators=1200, min_samples_split=5, 
            min_samples_leaf=2, max_features="sqrt", max_depth=90, bootstrap=True)
    trainmodel(rf, data, test)

# kaggle score: 0.79904
def trainRFOptimized2Model(data, test=None):
    # {'n_estimators': 10, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'max_depth': 95, 'bootstrap': True}
    rf = RandomForestClassifier(n_estimators=10, min_samples_split=3, 
            min_samples_leaf=3, max_features="sqrt", max_depth=95, bootstrap=True)
    trainmodel(rf, data, test)

# kaggle score: 0.785
def trainRFOptimized3Model(data, test=None):
    # {'n_estimators': 10, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'max_depth': 95, 'bootstrap': True}
    rf = RandomForestClassifier(n_estimators=10, min_samples_split=3, 
            min_samples_leaf=3, max_features="sqrt", max_depth=62, bootstrap=True)
    trainmodel(rf, data, test)

# kaggle score: 0.77, with smart age replacement, but somehow
# performing worse than the dumb age replacement
def trainRFOptimized4Model(data, test=None):
    rf = RandomForestClassifier(n_estimators=24, min_samples_split=7, 
            min_samples_leaf=3, max_features="sqrt", max_depth=50, bootstrap=True)
    trainmodel(rf, data, test)

# kaggle score: 0.78947
def trainVotingClassifier_LR_RF_SVC(data, test=None):
    lg = LogisticRegression()
    rf = RandomForestClassifier(oob_score=True)
    svc = SVC()
    master = VotingClassifier(estimators=[('lg', lg), ('rf', rf), ('svc', svc)], voting='hard')
    trainmodel(master, data, test)

# kaggle score: 0.79904
def trainXgboost(data, test=None):
    #{'colsample_bylevel': 1.0, 'colsample_bytree': 0.8, 
    #'max_delta_step': 1, 'max_depth': 9, 'n_estimators': 80, 'subsample': 0.6}
    model = xgboost.XGBRFClassifier(colsample_bylevel=1.0, colsample_bytree=0.8, max_delta_step=1, max_depth=10, n_estimators=80, subsample=0.6)
    trainmodel(model,data,test)

def trainmodels(data, test=None):
    #trainVotingClassifier_LR_RF_SVC(data, test)
    #trainRFOptimized2Model(data, test)
    #trainRFOptimized3Model(data, test)
    trainRFOptimized4Model(data, test)
    #trainXgboost(data, test)

def work():
    # program flow flags
    fMakePrediction = True
    fTuneParams = False
    fManualAnalysis = False

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train = transformData(train)
    test = transformData(test)

    if fManualAnalysis:
        #print(train.describe())
        grouped_train = train.groupby(['Sex','Pclass','Title'])
        grouped_median_train = grouped_train.median()
        grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
        print(grouped_median_train)
        print(train["Title"].value_counts())
        displayXNPlot(train, "Title vs Survived", 111, "Title", "Survived")

    if fMakePrediction:
        trainmodels(train, test)
        return

    if fTuneParams:
        featscols = ["SibSp","Parch","SexI","EmbarkedI","Pclass","AgeBin","FareBin","Title"]
        targetcol = "Survived"

        paramGridTest = {
            'n_estimators': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 
            'min_samples_split': [5, 6, 7, 8, 9], 
            'min_samples_leaf': [2, 3, 4], 
            'max_features': ["sqrt"], 
            'max_depth': [46, 48, 50, 52, 54], 
            'bootstrap': [True]
        }

        #grid = RandomizedSearchCV(RandomForestClassifier(), paramGridTest, cv=5, n_iter=100, verbose=2)
        grid = GridSearchCV(RandomForestClassifier(), paramGridTest, refit=True, verbose=2)
        #grid = GridSearchCV(xgboost.XGBRFClassifier(), paramGridTest, refit=True, verbose=0)
        grid.fit(train[featscols], train[targetcol])

        # print best parameter after tuning 
        print(grid.best_params_) 
        
        # print how our model looks after hyper-parameter tuning 
        print(grid.best_estimator_) 

        return

    #print(train[train["Fare"].isna()])
    #print(test[test["Fare"].isna()])
    
    #print(train[train["Name"].isna()])
    #print(test[test["Name"].isna()])

    #ds = train
    #ds = ds[ds["Embarked"] == 'S']
    #ds = ds[ds["Pclass"] == 3]
    #ds = ds["Fare"].median()
    #print(ds)

    #print(train["Ticket"].value_counts())

    #print(train.describe())
    #train["SibSpBin"] = putToBins(train, [-1, 0, 10], "SibSp")
    #displayXNPlot(train[["Survived", "SibSpBin"]], "SibSpBin vs Survived", 131, "SibSpBin", "Survived")

    #train["ParchBin"] = putToBins(train, [-1, 0, 10], "Parch")
    #displayXNPlot(train[["Survived", "ParchBin"]], "ParchBin vs Survived", 132, "ParchBin", "Survived")

    #train["Relatives"] = train["SibSp"] + train["Parch"]
    #train.loc[train['Relatives'] > 0, 'alone'] = 0
    #train.loc[train['Relatives'] == 0, 'alone'] = 1
    #train['alone'] = train['alone'].astype(int)
    #displayXNPlot(train[["Survived", "alone"]], "alone vs Survived", 133, "alone", "Survived")

    #print(train.info())

    #print(train[train["FareBin"].isna()].loc[:, ["PassengerId", "Fare"]])

    # drop passenger Id

    #displayAge(train, "Age", 111)

    # age categorized seems good

    #for aq in range(2, 6):
    #    train["AgeCat%d" % aq] = pd.qcut(train["Age"].dropna(), q=aq)
    #    displayXNPlot(train, "AgeCat%d vs Survived" % aq, 140 + aq - 1, "AgeCat%d" % aq, "Survived")
    #for aq in range(6, 10):
    #    train["AgeCat%d" % aq] = pd.qcut(train["Age"].dropna(), q=aq)
    #    displayXNPlot(train, "AgeCat%d vs Survived" % aq, 150 + aq - 5, "AgeCat%d" % aq, "Survived")
    #print(pd.qcut(train["Age"].dropna(), q=6))

    # fare seems good
    #train["FareCat"] = pd.qcut(train["Fare"].dropna(), q=7)
    #displayXNPlot(train, "FareCat vs Survived", 111, "FareCat", "Survived")
    #print(train["FareCat"])

    #train.info()

    # display top 8 rows of dataset.
    #print(train.head(8))

    # count null values per column
    #print(train.isna().sum())

    # to see all of the columns, maximizing the command line window size + reducing font size helps.
    #print(train.describe())

    #male = train[train["Sex"] == "male"]
    #female = train[train["Sex"] == "female"]

    #print(train["Age"].value_counts())

    #displayFareVsSurvivedPlot(train, "Fare", 111)
    #displayXNPlot(train[["Survived", "Pclass"]], "Pclass vs Survived", 111, "Pclass", "Survived")

    #displayAgeVsSurvivedPlot(male, "Male", 121)
    #displayAgeVsSurvivedPlot(female, "Female", 122)

    #displayXNPlot(train[["Survived", "Embarked"]], "Embarked vs Survived", 221, "Embarked", "Survived")
    #displayXNPlot(male[["Survived", "Embarked"]], "Embarked vs Male survived", 222, "Embarked", "Survived")
    #displayXNPlot(female[["Survived", "Embarked"]], "Embarked vs Female survived", 223, "Embarked", "Survived")


    #displayEmbarkedVsSurvivedPlot(male[male["Survived"] == 1], "Male Survived", 221)
    #displayEmbarkedVsSurvivedPlot(male[male["Survived"] == 0], "Male Dead", 222)
    #displayEmbarkedVsSurvivedPlot(female[female["Survived"] == 1], "Female Survived", 223)
    #displayEmbarkedVsSurvivedPlot(female[female["Survived"] == 0], "Female Dead", 224)

    plt.show()

def main():
    #sandbox()
    work()

if __name__ == '__main__':
    main()
