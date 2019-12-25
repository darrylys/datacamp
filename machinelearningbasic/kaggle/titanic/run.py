
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

    #SibSp,Parch, Fare
    # age is categorized to specific bins
    ageBins = [0, 18, 23, 28, 34, 44, 200]
    data["Age"] = data["Age"].fillna(29.7) # mean of age
    data["AgeBin"] = pd.cut(data["Age"], bins=ageBins, labels=[x for x in range(1, len(ageBins))]).astype(int)

    fareBins = [-1, 7.75, 8.05, 12.475, 19.258, 27.9, 56.929, 1000]
    # fare can be 0, NaN, huh.
    data["Fare"] = data["Fare"].fillna(8.05) # median fare of Embarked == S and Pclass == 3
    data["FareBin"] = pd.cut(data["Fare"], bins=fareBins, labels=[x for x in range(1, len(fareBins))])

    data["Relatives"] = data["SibSp"] + data["Parch"]
    data.loc[data['Relatives'] > 0, 'alone'] = 0
    data.loc[data['Relatives'] == 0, 'alone'] = 1
    data['alone'] = data['alone'].astype(int)

    # title is good predictor, but not improv AUC/Accuracy. Kaggle score: 0.75
    data["Title"] = data["Name"].map(lambda x : getTitleFromName(x))
    print(data["Title"].value_counts())

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

    featscols = ["SibSp","Parch","SexI","EmbarkedI","Pclass","AgeBin","FareBin","Title"]

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

def trainmodels(data, test=None):
    lg = LogisticRegression()
    rf = RandomForestClassifier(oob_score=True)
    svc = SVC()

    # adding VotingClassifier, increased the score a little bit
    # 0.78947
    master = VotingClassifier(estimators=[('lg', lg), ('rf', rf), ('svc', svc)], voting='hard')
    trainmodel(master, data, test)

def work():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train = transformData(train)
    test = transformData(test)
    trainmodels(train, test)

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
