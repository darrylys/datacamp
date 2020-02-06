
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score

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

titleMapBigOnly = {
    "mr": 1,
    "miss": 2,
    "mrs": 3,
    "master": 4
}

def getTitleCodeFromName(strName):
    return titleMap[getTitleFromName(strName)]

def getTitleFromNameBigOnly(strName):
    title = getTitleFromName(strName)
    if title in titleMapBigOnly:
        return title
    else:
        return 'other'

def getTitleFromName(strName):
    givenName = strName.split(",")[1]
    title = givenName.split(".")[0].lower().strip()
    return title

def showTitleVsSurvival(data):
    data["Title"] = data["Name"].apply(getTitleFromName)
    sns.countplot(x="Title", hue="Survived", data=data)
    plt.legend()
    plt.show()

def showAgeVsSurvival(data):
    """
    More females survive, most survival is in age ~20-40 or so.
    """
    sns.distplot(data[data["Survived"] == 1].Age.dropna(), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Age.dropna(), bins=20, kde=False, label="dead")
    plt.legend()
    plt.show()

def showInvAgeVsSurvival(data):
    """
    More females survive, most survival is in age ~20-40 or so.
    """
    sns.distplot(data[data["Survived"] == 1].Age.dropna().apply(lambda x: 1/(x+1)), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Age.dropna().apply(lambda x: 1/(x+1)), bins=20, kde=False, label="dead")
    plt.legend()
    plt.show()

def showAgeGenderVsSurvival(data):
    """
    Majority male not survived, majority female survived. Majority young kids, both M/F survived.
    """
    sns.swarmplot(data=data, x="Sex", y="Age", hue="Survived")
    plt.legend()
    plt.show()

def showFarePClassVsSurvived(data):
    """
    Majority Pclass 1 survived, and its fare is the highest. Others majority are not survived.
    """
    sns.swarmplot(data=data, x="Pclass", y="Fare", hue="Survived")
    plt.legend()
    plt.show()

def showEmbarkedFareVsSurvived(data):
    """
    Not very clear on relationship
    """
    sns.swarmplot(data=data, x="Embarked", y="Fare", hue="Survived")
    plt.legend()
    plt.show()

def showSiblingSpouseVsSurvived(data):
    """
    Fascinating, survival is most likely for SibSp <= 3 and Parch <= 3
    This could be investigated further. Is this correlated to Pclass?
    """
    sns.violinplot(data=data, x="SibSp", y="Parch", hue="Survived")
    plt.legend()
    plt.show()

def showFamilySizeVsSurvived(data):
    data["FamilySize"] = data["SibSp"] + data["Parch"]
    sns.countplot(x="FamilySize", hue="Survived",  data=data)
    plt.legend()
    plt.show()

def showSibSpParchPclass(data):
    """
    As expected, the higher SibSp and Parch are in lower Pclass and they don't survive, 
    supported the conclusion from Pclass vs survivability earlier.
    """
    sns.scatterplot(data=data, x="SibSp", y="Parch", hue="Pclass")
    plt.legend()
    plt.show()

def showCabinVsSurvived(data):
    """
    Seems promising for B,C,D and E cabins, but not much elsewhere.
    The problem for cabin is that ~70% of the data is MISSING!
    For missing data, majority of them not survived.
    """
    data2 = data[["Cabin", "Survived"]].fillna('X')
    data2["CabinFL"] = data2["Cabin"].apply(lambda x: x[0])
    sns.countplot(x="CabinFL", hue="Survived", data=data2)
    plt.legend()
    plt.show()

def showCabinVsFare(data):
    """
    Cabin BCDE seems to be the most expensive.
    Cabin AFTG should be combined to one label: Other.
    """
    data2 = data[["Cabin", "Fare", "Survived"]].dropna()
    data2["CabinFL"] = data2["Cabin"].apply(lambda x: x[0])
    sns.swarmplot(data=data2, x="CabinFL", y="Fare")
    plt.legend()
    plt.show()

def showFareDist(data):
    data["FareLog"] = data["Fare"].apply(lambda x : math.log(x+1))

    plt.subplot(121)
    plt.tight_layout()
    sns.violinplot(data=data, x="Survived", y="Fare")
    plt.title("Survived vs Fare")

    plt.subplot(122)
    plt.tight_layout()
    sns.violinplot(data=data, x="Survived", y="FareLog")
    plt.title("Survived vs FareLog")

    plt.legend()
    plt.show()

def stdScale(X_train, X_test, columnName):
    fsStdScaler = StandardScaler()
    fsStdScaler = fsStdScaler.fit(X_train[[columnName]])
    X_train.loc[:,[columnName]] = fsStdScaler.transform(X_train[[columnName]])
    X_test.loc[:,[columnName]] = fsStdScaler.transform(X_test[[columnName]])
    return X_train, X_test

def rfattempt1_transform(X_train, X_test):
    # encode Sex
    sexEncoder = LabelEncoder()
    sexEncoder = sexEncoder.fit(X_train["Sex"])
    X_train.loc[:,["Sex"]] = sexEncoder.transform(X_train["Sex"])
    X_test.loc[:,["Sex"]] = sexEncoder.transform(X_test["Sex"])

    # fill age with median
    medianAge = np.median(X_train["Age"].dropna())
    X_train.loc[:,["Age"]] = X_train["Age"].fillna(medianAge)
    X_test.loc[:,["Age"]] = X_test["Age"].fillna(medianAge)
    X_train, X_test = stdScale(X_train, X_test, "Age")

    # create new feature, Family Size 
    def mxs(row):
        return row.SibSp + row.Parch
    X_train["FamilySize"] = X_train.apply(mxs, axis=1)
    X_test["FamilySize"] = X_test.apply(mxs, axis=1)
    X_train, X_test = stdScale(X_train, X_test, "FamilySize")
    X_train, X_test = stdScale(X_train, X_test, "SibSp")
    X_train, X_test = stdScale(X_train, X_test, "Parch")

    #X_train = X_train.drop(["SibSp", "Parch"], axis=1)
    #X_test = X_test.drop(["SibSp", "Parch"], axis=1)

    # Fare
    medianFare = np.median(X_train["Fare"].dropna())
    X_train.loc[:,["Fare"]] = X_train["Fare"].fillna(medianFare)
    X_test.loc[:,["Fare"]] = X_test["Fare"].fillna(medianFare)
    X_train, X_test = stdScale(X_train, X_test, "Fare")

    # Cabin
    # Cabin reduces score. Since it's missing ~70%, just remove this.
    #cabinCode = {
    #    "A": 1,
    #    "B": 2,
    #    "C": 3,
    #    "D": 4,
    #    "E": 5,
    #    "F": 6,
    #    "G": 7,
    #    "T": 8,
    #    "U": 9
    #}
    #X_train.loc[:,["Cabin"]] = X_train["Cabin"].fillna("U")
    #X_train.loc[:,["Cabin"]] = X_train["Cabin"].apply(lambda x: cabinCode[x[0]])
    
    #X_test.loc[:,["Cabin"]] = X_test["Cabin"].fillna("U")
    #X_test.loc[:,["Cabin"]] = X_test["Cabin"].apply(lambda x: cabinCode[x[0]])

    X_train = X_train.drop(["Cabin"], axis=1)
    X_test = X_test.drop(["Cabin"], axis=1)

    # embarked
    X_train.loc[:,["Embarked"]] = X_train["Embarked"].fillna("S")
    #embEncoder = LabelEncoder()
    #embEncoder = embEncoder.fit(X_train["Embarked"])
    #X_train.loc[:,["Embarked"]] = embEncoder.transform(X_train["Embarked"])
    #X_test.loc[:,["Embarked"]] = embEncoder.transform(X_test["Embarked"].fillna("S"))

    # add a little bit (0.76)
    X_train["Title"] = X_train["Name"].apply(getTitleFromNameBigOnly)
    X_test["Title"] = X_test["Name"].apply(getTitleFromNameBigOnly)

    #df.join(pd.get_dummies(df[['A', 'B']], prefix=['col1', 'col2']))
    X_train = X_train.join(pd.get_dummies(X_train[["Embarked", "Title"]], prefix=["Embarked", "Title"]))
    X_test = X_test.join(pd.get_dummies(X_test[["Embarked", "Title"]], prefix=["Embarked", "Title"]))

    X_train = X_train.drop(["Name", "Embarked", "Title"], axis=1)
    X_test = X_test.drop(["Name", "Embarked", "Title"], axis=1)

    return X_train, X_test

def trainSVC(X_train, y_train, X_test):
    model = SVC()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def trainLogReg(X_train, y_train, X_test):
    model = LogisticRegression(penalty='l2', C=1.0)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def trainRandomForest(X_train, y_train, X_test):
    model = RandomForestClassifier()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)

    return model, y_pred

def trainVotingClassifier(X_train, y_train, X_test):
    lg = LogisticRegression(penalty='l2', C=1.0)
    rf = RandomForestClassifier()
    svc = SVC()
    model = VotingClassifier(estimators=[('lg', lg), ('rf', rf), ('svc', svc)], voting='hard')

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def trainML(X_train, y_train, X_test):
    return trainSVC(X_train, y_train, X_test)

def rfattempt1(df_train, df_forSubs):
    kf = KFold(n_splits=5)

    oriFeatures = df_train.columns.drop(["PassengerId","Survived","Ticket"])
    X = df_train[oriFeatures]
    y = df_train["Survived"]

    aucList = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = rfattempt1_transform(X_train, X_test)

        model, y_pred = trainML(X_train, y_train, X_test)

        auc = roc_auc_score(y_test, y_pred)
        aucList.append(auc)
        
    print("Avg auc: {}".format(np.mean(aucList)))

    print("Predicting result in test.csv")
    X_p = df_forSubs[oriFeatures]

    X_train, X_test = rfattempt1_transform(X, X_p)
    #print(X_train.describe(include='all'))
    #print(X_test.describe(include='all'))

    model, y_p = trainML(X_train, y, X_test)

    df_forSubs["Survived"] = y_p
    df_forSubs[["PassengerId", "Survived"]].to_csv("test.predicted.csv", index=False)

    print("Done")

def showTitles(x1, x2):
    x1title = x1["Name"].apply(getTitleFromName)
    x2title = x2["Name"].apply(getTitleFromName)
    print(x1title.value_counts())
    print(x2title.value_counts())
    #print(pd.Series(titles).value_counts())


def main():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    oriFeatures = df_train.columns.drop(["PassengerId","Survived"])

    #print(df_train["Name"].apply(getTitleFromName).value_counts())
    #print(df_test["Name"].apply(getTitleFromName).value_counts())

    #print("CabinFL: {}".format(df_train["Cabin"].dropna().apply(lambda x: x[0]).value_counts()))
    #print(df_train[oriFeatures].describe(include='all'))
    #print(df_train["Embarked"].value_counts())
    #print(df_train["Cabin"].value_counts())
    #print(df_train["Pclass"].value_counts())
    #showAgeVsSurvival(df_train)
    #showInvAgeVsSurvival(df_train)
    #showAgeGenderVsSurvival(df_train)
    #showFareVsPClass(df_train)
    #showEmbarkedFareVsSurvived(df_train)
    #showSiblingSpouseVsSurvived(df_train)
    #showFamilySizeVsSurvived(df_train)
    #showSibSpParchPclass(df_train)
    #print("Unique cabin numbers: {}".format(np.unique(df_train["Cabin"].dropna())))
    #showCabinVsSurvived(df_train)
    #showCabinVsFare(df_train)
    #showTitleVsSurvival(df_train)
    #showFareDist(df_train)
    rfattempt1(df_train, df_test)
    #showTitles(df_train, df_test)

    #df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
    #               'C': [1, 2, 3]})
    #df = df.join(pd.get_dummies(df[['A', 'B']], prefix=['col1', 'col2']))
    #print(df)
    

if __name__ == '__main__':
    main()
