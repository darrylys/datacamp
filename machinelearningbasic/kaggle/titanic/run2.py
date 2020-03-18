
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
from sklearn.svm import SVC, LinearSVC

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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

titleSimplificationMap = {
    "mr": "mr",
    "miss": "miss",
    "mrs": "mrs",
    "master": "master",
    "rev": "mr",
    "col": "mr",
    "dona": "mrs",
    "dr": "mr",
    "ms": "mrs",
    "mlle": "miss",
    "major": "mr",
    "don": "mr",
    "mme": "mrs",
    "capt": "mr",
    "jonkheer": "mr",
    "sir": "mr",
    "the countess": "mrs",
    "lady": "mrs"
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
    return titleSimplificationMap[title]

    #if title in titleMapBigOnly:
    #    return title
    #else:
    #    return 'other'

def getTitleFromName(strName):
    givenName = strName.split(",")[1]
    title = givenName.split(".")[0].lower().strip()
    return title

def getFamilyName(strName):
    familyName = strName.split(",")[0]
    return familyName.strip().lower()

def showFamilyCountVsSurvival(data):
    data.loc[:, "FamilyName"] = data["Name"].apply(getFamilyName)
    vc = data["FamilyName"].value_counts()
    familyCounts = {}
    for k, v in zip(vc.index.values, vc.values):
        familyCounts[k] = v
    
    data.loc[:, "FamilyNameCount"] = data["FamilyName"].map(familyCounts)
    sns.countplot(x="FamilyNameCount", hue="Survived", data=data)
    plt.legend()
    plt.show()

def showTitleVsSurvival(data):
    data["Title"] = data["Name"].apply(getTitleFromName)
    sns.countplot(x="Title", hue="Survived", data=data)
    plt.legend()
    plt.show()

def showAgeVsTitleCorr(data):
    data.loc[:,"Title"] = data["Name"].apply(getTitleFromName)
    titles = np.unique(data["Title"])
    for title in titles:
        titledata = data[data["Title"] == title].Age.dropna()
        lens = titledata.shape[0]
        if lens >= 10:
            sns.distplot(titledata, bins=20, kde=False, label=title)
    plt.legend()
    plt.show()

def showAgeVsSurvival(data):
    """
    More females survive, most survival is in age ~20-40 or so.
    Distribution seems right-er
    """
    sns.distplot(data[data["Survived"] == 1].Age.dropna(), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Age.dropna(), bins=20, kde=False, label="dead")
    plt.legend()
    plt.show()

def showAgeXPclassVsSurvival(data):
    data["Age"] = data["Age"].dropna()
    data["Pclass"] = data["Pclass"].dropna()

    data["AgeXPclass"] = data["Age"] * data["Pclass"]
    data["AgeXPclass"] = data["AgeXPclass"].apply(lambda x : math.sqrt(x))

    sns.distplot(data[data["Survived"] == 1].AgeXPclass.dropna(), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].AgeXPclass.dropna(), bins=20, kde=False, label="dead")
    plt.legend()
    plt.show()

def putToBins(data, bins, colName):
    return pd.cut(data[colName], bins=bins, 
            labels=[x for x in range(1, len(bins))])

def showAgeQcutVsSurvival(data):
    data.loc[:, "AgeQ10"], bin_edges = pd.qcut(data["Age"], q=10, retbins=True)
    print(bin_edges)
    print(data["AgeQ10"].value_counts())
    sns.countplot(x="AgeQ10", hue="Survived", data=data)
    plt.legend()
    plt.show()

def concat_df(train, test):
    return pd.concat([train, test], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

def obtainBinningForFareAndAge(train, test):
    all_data = concat_df(train, test)
    _, bins = pd.qcut(all_data["Age"], q=10, retbins=True)
    print("Age: {}".format(bins))

    _, bins = pd.qcut(all_data["Fare"], q=13, retbins=True)
    print("Fare: {}".format(bins))

def showFareQcutVsSurvival(data):
    data.loc[:, "FareQ13"] = pd.qcut(data["Fare"], q=13)
    sns.countplot(x="FareQ13", hue="Survived", data=data)
    plt.legend()
    plt.show()

def showAgeOntologyVsSurvival(data):
    """
    Loosely following this:
    https://www.researchgate.net/figure/Age-range-classes-defined-by-the-Age-Ontology-Age-range-classes-were-generally-defined_fig1_271840217

    """
    data["Age"] = data["Age"].dropna()
    data["AgeBin"] = putToBins(data, [0, 11, 19, 23, 28, 33, 41, 65, 200], "Age")
    print(data["AgeBin"])
    print(data["AgeBin"].value_counts())
    data["Title"] = data["Name"].apply(getTitleFromNameBigOnly)
    medianPerTitle(data)
    sns.countplot(x="AgeBin", hue="Survived", data=data)
    plt.legend()
    plt.show()

def showAgeCAOVsSurvival(data):
    """
    Loosely following this:
    https://www.researchgate.net/figure/Age-range-classes-defined-by-the-Age-Ontology-Age-range-classes-were-generally-defined_fig1_271840217

    """
    data.loc[:,"Age"] = data["Age"].dropna()
    data.loc[:,"AgeBin"] = putToBins(data, [0, 11, 19, 65, 200], "Age")
    sns.countplot(x="AgeBin", hue="Survived", data=data)
    plt.legend()
    plt.show()

def showInvAgeVsSurvival(data):
    """
    Too much age on the left
    """
    sns.distplot(data[data["Survived"] == 1].Age.dropna().apply(lambda x: 1/(x+1)), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Age.dropna().apply(lambda x: 1/(x+1)), bins=20, kde=False, label="dead")
    plt.legend()
    plt.show()

def showLogAgeVsSurvival(data):
    """
    Too much age on the right
    """
    sns.distplot(data[data["Survived"] == 1].Age.dropna().apply(lambda x: math.log(x+1)), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Age.dropna().apply(lambda x: math.log(x+1)), bins=20, kde=False, label="dead")
    plt.legend()
    plt.show()

def showSqrtAgeVsSurvival(data):
    """
    Too much age on the center??
    """
    sns.distplot(data[data["Survived"] == 1].Age.dropna().apply(lambda x: math.sqrt(x)), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Age.dropna().apply(lambda x: math.sqrt(x)), bins=20, kde=False, label="dead")
    plt.legend()
    plt.show()

def showFareVsSurvival(data):
    """
    Too much age on the center??
    """
    sns.distplot(data[data["Survived"] == 1].Fare.dropna().apply(lambda x: x), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Fare.dropna().apply(lambda x: x), bins=20, kde=False, label="dead")
    plt.title("Fare")
    plt.legend()
    plt.show()

def showLogFareVsSurvival(data):
    """
    Too much age on the center??
    """
    sns.distplot(data[data["Survived"] == 1].Fare.dropna().apply(lambda x: math.log(x+1)), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Fare.dropna().apply(lambda x: math.log(x+1)), bins=20, kde=False, label="dead")
    plt.title("Log Fare")
    plt.legend()
    plt.show()

def showSqrtFareVsSurvival(data):
    """
    Too much age on the center??
    """
    sns.distplot(data[data["Survived"] == 1].Fare.dropna().apply(lambda x: math.sqrt(x)), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Fare.dropna().apply(lambda x: math.sqrt(x)), bins=20, kde=False, label="dead")
    plt.title("Sqrt Fare")
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
    plt.title("FamilySize Survived")
    plt.legend()
    plt.show()

def showFamilySize2VsSurvived(data):
    data["FamilySize"] = data["SibSp"] + 10 * data["Parch"]
    sns.countplot(x="FamilySize", hue="Survived",  data=data)
    plt.title("FamilySize 10 Survived")
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
    data2.loc[:,"CabinFL"] = data2["Cabin"].apply(lambda x: x[0])
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

def showFareVsFare_Person(data):
    th = {}
    def thf(x):
        if x in th:
            th[x] += 1
        else:
            th[x] = 1
        return x

    data.loc[:,"Ticket"].apply(thf)
    data.loc[:,"SameTicketsN"] = data["Ticket"].apply(lambda x : th[x])

    lg1 = lambda x : math.log(x+1)
    plt.subplot(121)
    plt.tight_layout()
    sns.distplot(data[data["Survived"] == 1].Fare.dropna().apply(lg1), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Fare.dropna().apply(lg1), bins=20, kde=False, label="dead")
    plt.legend()

    plt.subplot(122)
    plt.tight_layout()
    data.loc[:,"Fare_Person"] = data["Fare"] / data["SameTicketsN"]
    sns.distplot(data[data["Survived"] == 1].Fare_Person.dropna().apply(lg1), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Fare_Person.dropna().apply(lg1), bins=20, kde=False, label="dead")
    plt.legend()

    plt.show()

def plotTickets(data):
    """
    People who buys same ticket (5,6,7) are mostly dead
    If ticket is shared / bought by 2 or more people, number of survive / dead is near 50:50
    """
    th = {}
    def thf(x):
        if x in th:
            th[x] += 1
        else:
            th[x] = 1
        return x

    data.loc[:,"Ticket"].apply(thf)
    data.loc[:,"SameTicketsN"] = data["Ticket"].apply(lambda x : th[x])
    data.loc[:,"IsSharedTicket"] = data["SameTicketsN"].apply(lambda x: 1 if x >= 2 else 0)

    plt.subplot(221)
    plt.tight_layout()
    sns.countplot(x="SameTicketsN", hue="Survived", data = data)
    plt.legend()

    plt.subplot(222)
    plt.tight_layout()
    sns.countplot(x="IsSharedTicket", hue="Survived", data = data)
    plt.legend()

    plt.subplot(223)
    plt.tight_layout()
    data.loc[:,"FamilySize"] = data["SibSp"] + data["Parch"]
    sns.scatterplot(x="SameTicketsN", y="FamilySize", hue="Survived", data=data)
    plt.legend()

    plt.subplot(224)
    plt.tight_layout()
    data.loc[:,"Fare_Person"] = data["Fare"] / data["SameTicketsN"]
    sns.distplot(data[data["Survived"] == 1].Fare_Person.dropna().apply(lambda x: x), bins=20, kde=False, label="survived")
    sns.distplot(data[data["Survived"] == 0].Fare_Person.dropna().apply(lambda x: x), bins=20, kde=False, label="dead")
    plt.legend()

    plt.show()
    

def select_features_l1(X, y):
    """ Return selected features using logistic regression with an L1 penalty """
    logistic = LogisticRegression(C=1.0, penalty='l1', solver='liblinear').fit(X, y)
    model = SelectFromModel(logistic, prefit=True)
    
    X_new = model.transform(X)
    selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                index=X.index,
                                columns=X.columns)

    # Dropped columns have values of all 0s, keep other columns 
    selected_columns = selected_features.columns[selected_features.var() != 0]
    return selected_columns

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
    X_train.loc[:,columnName] = fsStdScaler.transform(X_train[[columnName]])
    X_test.loc[:,columnName] = fsStdScaler.transform(X_test[[columnName]])
    return X_train, X_test

def labelEncode(X_train, X_test, columnName):
    enc = LabelEncoder()
    enc = enc.fit(X_train[columnName])
    X_train.loc[:,columnName] = enc.transform(X_train[columnName])
    X_test.loc[:,columnName] = enc.transform(X_test[columnName])
    return X_train, X_test

def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    #return df[by].map(smooth)
    return smooth

def medianPerTitle(X_train):
    hm = {}
    for entry in X_train[["Title", "Age"]].dropna().values:
        title = entry[0]
        age = entry[1]
        if title in hm:
            hm[title].append(age)
        else:
            hm[title] = [age]
    
    for title in hm:
        hm[title] = np.median(hm[title])
    
    #print(hm)

    return hm

def rfattempt1_transform(X_train, X_test, y_train, y_test=None):
    # encode Sex
    X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, "Sex")
    
    # add a little bit (0.76)
    # for obtaining the median of age, try add more precision to titles
    X_train.loc[:,"Title"] = X_train["Name"].apply(getTitleFromNameBigOnly)
    X_test.loc[:,"Title"] = X_test["Name"].apply(getTitleFromNameBigOnly)
    #X_train, X_test = labelEncode(X_train, X_test, "Title")

    # YAAY!, with smarter median, increased to 0.794
    # with title
    medianPerTitleMap = medianPerTitle(X_train)
    #medianAge = np.median(X_train["Age"].dropna())
    def smarterMedianProbablyIdk(row):
        if np.isnan(row.Age):
            return medianPerTitleMap[row.Title]
        else:
            return row.Age
    X_train.loc[:,"Age"] = X_train.apply(smarterMedianProbablyIdk, axis=1)
    X_test.loc[:,"Age"] = X_test.apply(smarterMedianProbablyIdk, axis=1)

    age_bins = [-1,14,19,22,25,28,31,36,42,50,80]
    X_train.loc[:,"AgeQ10"] = pd.cut(X_train["Age"], bins=age_bins)
    X_test.loc[:,"AgeQ10"] = pd.cut(X_test["Age"], bins=age_bins)
    X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, "AgeQ10")
    
    X_train = X_train.drop(["Age"], axis=1)
    X_test = X_test.drop(["Age"], axis=1)

    # create new feature, Family Size 
    def mxs(row):
        return 1 + row.SibSp + row.Parch
    X_train.loc[:,"FamilySize"] = X_train.apply(mxs, axis=1)
    X_test.loc[:,"FamilySize"] = X_test.apply(mxs, axis=1)

    family_size_bin = [-1, 1, 4, 7, 100]
    X_train.loc[:,"FamilySizeBin"] = pd.cut(X_train["FamilySize"], bins=family_size_bin)
    X_test.loc[:,"FamilySizeBin"] = pd.cut(X_test["FamilySize"], bins=family_size_bin)
    X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, "FamilySizeBin")
    #X_train, X_test = stdScale(X_train, X_test, "FamilySize")

    X_train, X_test = stdScale(X_train, X_test, "SibSp")
    X_train, X_test = stdScale(X_train, X_test, "Parch")

    X_train = X_train.drop(["FamilySize"], axis=1)
    X_test = X_test.drop(["FamilySize"], axis=1)

    # Cabin
    cabinCode = {
        "A": "PABC",
        "B": "PABC",
        "C": "PABC",
        "D": "PDE",
        "E": "PDE",
        "F": "PFG",
        "G": "PFG",
        "T": "PABC",
        "U": "PU"
    }
    X_train.loc[:,"Cabin"] = X_train["Cabin"].fillna("U")
    X_train.loc[:,"Cabin"] = X_train["Cabin"].apply(lambda x: cabinCode[x[0]])
    X_test.loc[:,"Cabin"] = X_test["Cabin"].fillna("U")
    X_test.loc[:,"Cabin"] = X_test["Cabin"].apply(lambda x: cabinCode[x[0]])

    # OHE embarked, title
    modeEmbarked = 'S'
    X_train.loc[:,"Embarked"] = X_train["Embarked"].fillna(modeEmbarked)
    X_test.loc[:,"Embarked"] = X_test["Embarked"].fillna(modeEmbarked)

    X_train.loc[:,"Pclass"] = X_train["Pclass"].map({1: "High", 2: "Middle", 3: "Low"})
    X_test.loc[:,"Pclass"] = X_test["Pclass"].map({1: "High", 2: "Middle", 3: "Low"})

    # get_dummies cannot work if data is numeric.
    dum = ["Embarked", "Title", "Pclass", "Cabin"]
    #for duf in dum:
    #    X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, duf)
    X_train = X_train.join(pd.get_dummies(X_train[dum], prefix=dum))
    X_test = X_test.join(pd.get_dummies(X_test[dum], prefix=dum))

    # analyze tickets
    # SameTicketsN Doesn't influence the final result.
    th = {}
    def thf(x):
        if x in th:
            th[x] += 1
        else:
            th[x] = 1
        return x

    X_train["Ticket"].apply(thf)
    X_test["Ticket"].apply(thf)
    X_train.loc[:,"SameTicketsN"] = X_train["Ticket"].apply(lambda x : th[x])
    X_test.loc[:,"SameTicketsN"] = X_test["Ticket"].apply(lambda x : th[x])

    # Fare only has 1 missing data, filling with smart median is not worth it
    medianFare = X_train["Fare"].median()
    X_train.loc[:,"Fare"] = X_train["Fare"].fillna(medianFare)
    X_test.loc[:,"Fare"] = X_test["Fare"].fillna(medianFare)

    fare_bins = [-1,7.25,7.75,7.8958,8.05,10.5,13.,15.7417,23.25,26.55,34.15703077,56.4958,83.475,512.3292]
    X_train.loc[:,"FareQ13"] = pd.cut(X_train["Fare"], bins=fare_bins)
    X_test.loc[:,"FareQ13"] = pd.cut(X_test["Fare"], bins=fare_bins)
    X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, "FareQ13")

    #Xy_train = X_train.join(y_train)
    #smoothed = calc_smooth_mean(Xy_train, by="SameTicketsN", on='Survived', m=Xy_train.shape[0])
    #X_train.loc[:,'SameTicketsN'] = X_train["SameTicketsN"].map(smoothed)
    #X_test.loc[:,'SameTicketsN'] = X_test["SameTicketsN"].map(smoothed)
    X_train, X_test = stdScale(X_train, X_test, "SameTicketsN")

    X_train = X_train.drop(["Name", "Ticket", "Fare", "Embarked", "Title", "Pclass", "Cabin"], axis=1)
    X_test = X_test.drop(["Name", "Ticket", "Fare", "Embarked", "Title", "Pclass", "Cabin"], axis=1)

    return X_train, X_test

def smoothMeanLabelling(X_train, y_train, X_test, featureColumn, targetColumn="Survived"):
    Xy_train = X_train.join(y_train)
    smoothed = calc_smooth_mean(Xy_train, by=featureColumn, on=targetColumn, m=Xy_train.shape[0])
    X_train.loc[:,featureColumn] = X_train[featureColumn].map(smoothed)
    X_test.loc[:,featureColumn] = X_test[featureColumn].map(smoothed)
    return X_train, X_test

def rfattempt2_forRandomForest(X_train, X_test, y_train, y_test=None):
    # encode Sex
    X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, "Sex")
    
    # add a little bit (0.76)
    # for obtaining the median of age, try add more precision to titles
    X_train.loc[:,"Title"] = X_train["Name"].apply(getTitleFromNameBigOnly)
    X_test.loc[:,"Title"] = X_test["Name"].apply(getTitleFromNameBigOnly)
    #X_train, X_test = labelEncode(X_train, X_test, "Title")

    # YAAY!, with smarter median, increased to 0.794
    # with title
    medianPerTitleMap = medianPerTitle(X_train)
    #medianAge = np.median(X_train["Age"].dropna())
    def smarterMedianProbablyIdk(row):
        if np.isnan(row.Age):
            return medianPerTitleMap[row.Title]
        else:
            return row.Age
    X_train.loc[:,"Age"] = X_train.apply(smarterMedianProbablyIdk, axis=1)
    X_test.loc[:,"Age"] = X_test.apply(smarterMedianProbablyIdk, axis=1)

    age_bins = [-1,14,19,22,25,28,31,36,42,50,80]
    X_train.loc[:,"AgeQ10"] = pd.cut(X_train["Age"], bins=age_bins)
    X_test.loc[:,"AgeQ10"] = pd.cut(X_test["Age"], bins=age_bins)
    X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, "AgeQ10")
    
    X_train = X_train.drop(["Age"], axis=1)
    X_test = X_test.drop(["Age"], axis=1)

    # create new feature, Family Size 
    def mxs(row):
        return row.SibSp + row.Parch
    X_train.loc[:,"FamilySize"] = X_train.apply(mxs, axis=1)
    X_test.loc[:,"FamilySize"] = X_test.apply(mxs, axis=1)
    X_train, X_test = stdScale(X_train, X_test, "FamilySize")

    X_train, X_test = stdScale(X_train, X_test, "SibSp")
    X_train, X_test = stdScale(X_train, X_test, "Parch")

    #X_train = X_train.drop(["SibSp", "Parch"], axis=1)
    #X_test = X_test.drop(["SibSp", "Parch"], axis=1)

    # Cabin
    cabinCode = {
        "A": "PABC",
        "B": "PABC",
        "C": "PABC",
        "D": "PDE",
        "E": "PDE",
        "F": "PFG",
        "G": "PFG",
        "T": "PABC",
        "U": "PU"
    }
    X_train.loc[:,"Cabin"] = X_train["Cabin"].fillna("U")
    X_train.loc[:,"Cabin"] = X_train["Cabin"].apply(lambda x: cabinCode[x[0]])
    X_test.loc[:,"Cabin"] = X_test["Cabin"].fillna("U")
    X_test.loc[:,"Cabin"] = X_test["Cabin"].apply(lambda x: cabinCode[x[0]])

    # OHE embarked, title
    modeEmbarked = 'S'
    X_train.loc[:,"Embarked"] = X_train["Embarked"].fillna(modeEmbarked)
    X_test.loc[:,"Embarked"] = X_test["Embarked"].fillna(modeEmbarked)
    print(X_train[X_train["Embarked"].isnull()])

    X_train.loc[:,"Pclass"] = X_train["Pclass"].map({1: "High", 2: "Middle", 3: "Low"})
    X_test.loc[:,"Pclass"] = X_test["Pclass"].map({1: "High", 2: "Middle", 3: "Low"})

    # get_dummies cannot work if data is numeric.
    dum = ["Embarked", "Title", "Pclass", "Cabin"]
    for duf in dum:
        print(duf)
        X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, duf)

    # analyze tickets
    # SameTicketsN Doesn't influence the final result.
    th = {}
    def thf(x):
        if x in th:
            th[x] += 1
        else:
            th[x] = 1
        return x

    X_train["Ticket"].apply(thf)
    X_test["Ticket"].apply(thf)
    X_train.loc[:,"SameTicketsN"] = X_train["Ticket"].apply(lambda x : th[x])
    X_test.loc[:,"SameTicketsN"] = X_test["Ticket"].apply(lambda x : th[x])

    # Fare only has 1 missing data, filling with smart median is not worth it
    medianFare = X_train["Fare"].median()
    X_train.loc[:,"Fare"] = X_train["Fare"].fillna(medianFare)
    X_test.loc[:,"Fare"] = X_test["Fare"].fillna(medianFare)

    fare_bins = [-1,7.25,7.75,7.8958,8.05,10.5,13.,15.7417,23.25,26.55,34.15703077,56.4958,83.475,512.3292]
    X_train.loc[:,"FareQ13"] = pd.cut(X_train["Fare"], bins=fare_bins)
    X_test.loc[:,"FareQ13"] = pd.cut(X_test["Fare"], bins=fare_bins)
    X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, "FareQ13")

    #Xy_train = X_train.join(y_train)
    #smoothed = calc_smooth_mean(Xy_train, by="SameTicketsN", on='Survived', m=Xy_train.shape[0])
    #X_train.loc[:,'SameTicketsN'] = X_train["SameTicketsN"].map(smoothed)
    #X_test.loc[:,'SameTicketsN'] = X_test["SameTicketsN"].map(smoothed)
    X_train, X_test = stdScale(X_train, X_test, "SameTicketsN")

    X_train = X_train.drop(["Name", "Ticket", "Fare"], axis=1)
    X_test = X_test.drop(["Name", "Ticket", "Fare"], axis=1)

    return X_train, X_test

# score: 0.779
def trainLinearSVC(X_train, y_train, X_test):
    model = SVC(kernel='linear')
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("# SuppVec / len: {}".format(np.sum(model.n_support_) / len(y_train))) 

    print(f"Feature importance:")
    coefs = model.coef_.tolist()[0]
    feats = X_train.columns.values
    bc = sorted(zip(coefs, feats), key=lambda x: abs(x[0]), reverse=True)
    print(f"sorted zip(coef, feats)={bc}")
    
    return model, y_pred

# score: 0.789 - 0.794
def trainSVC(X_train, y_train, X_test):
    model = SVC(C=1.0, degree=3, gamma='scale', kernel='rbf')
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("# SuppVec / len: {}".format(np.sum(model.n_support_) / len(y_train))) 
    return model, y_pred

# score:0.770
def trainLogReg(X_train, y_train, X_test):
    model = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

# score: 0.720 - 0.75
def trainRandomForest(X_train, y_train, X_test):
    model = RandomForestClassifier()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)

    return model, y_pred

# score: 0.77
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

    oriFeatures = df_train.columns.drop(["PassengerId", "Survived"])
    X = df_train[oriFeatures].copy()
    y = df_train["Survived"].copy()

    aucList = []
    accList = []
    f1sList = []

    for train_idx, test_idx in kf.split(X):
        #print("Begin splitting data to train/test pair")

        # read the following to know why .copy() is required!
        # https://www.dataquest.io/blog/settingwithcopywarning/
        X_train, X_test = X.iloc[train_idx,:].copy(), X.iloc[test_idx,:].copy()
        y_train, y_test = y[train_idx].copy(), y[test_idx].copy()

        #print("Start transforming data")

        X_train, X_test = rfattempt1_transform(X_train, X_test, y_train, y_test)
        #X_train, X_test = rfattempt2_forRandomForest(X_train, X_test, y_train, y_test)

        print("Transformed Train/Test dataset: ")
        print(X_train.head(5))
        print(X_test.head(5))
        model, y_pred = trainML(X_train, y_train, X_test)

        aucList.append(roc_auc_score(y_test, y_pred))
        accList.append(accuracy_score(y_test, y_pred))
        f1sList.append(f1_score(y_test, y_pred))
        
    print("Avg auc: {}".format(np.mean(aucList)))
    print("Avg acc: {}".format(np.mean(accList)))
    print("Avg F1-score: {}".format(np.mean(f1sList)))

    print("Predicting result in test.csv")
    X_p = df_forSubs[oriFeatures].copy()

    X_train, X_test = rfattempt1_transform(X, X_p, y)
    #X_train, X_test = rfattempt2_forRandomForest(X, X_p, y)
    #print(X_train.describe(include='all'))
    #print(X_test.describe(include='all'))

    model, y_p = trainML(X_train, y, X_test)

    df_forSubs.loc[:,"Survived"] = y_p
    df_forSubs[["PassengerId", "Survived"]].to_csv("test.predicted.csv", index=False)

    print("Done")

def showTitles(x1, x2):
    x1title = x1["Name"].apply(getTitleFromName)
    x2title = x2["Name"].apply(getTitleFromName)
    print(x1title.value_counts())
    print(x2title.value_counts())
    #print(pd.Series(titles).value_counts())

def filterFeaturesQuestionable(df_train):
    kf = KFold(n_splits=5)

    oriFeatures = df_train.columns.drop(["PassengerId","Survived"])
    X = df_train[oriFeatures]
    y = df_train["Survived"]

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = rfattempt1_transform(X_train, X_test, y_train, y_test)

        selected_features = select_features_l1(X_train, y_train)

        print("selected features based on LogReg L1: {}".format(selected_features))

        break

def searchParamsMaybeIllegal(df_train):
    kf = KFold(n_splits=5)

    oriFeatures = df_train.columns.drop(["PassengerId","Survived"])
    X = df_train[oriFeatures]
    y = df_train["Survived"]

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = rfattempt1_transform(X_train, X_test, y_train, y_test)

        paramGridTest = {
            'C': [0.001, 0.01, 0.1, 1, 10], 
            'gamma': [0.001, 0.01, 0.1, 1, 'scale', 10], 
            'degree': [2, 3, 4],
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'coef0': [0.001, 0.01, 0.1, 0, 1]
        }

        grid = GridSearchCV(SVC(), paramGridTest, refit=True, verbose=2)
        grid.fit(X_train, y_train)

        # print best parameter after tuning 
        print(grid.best_params_) 
        
        # print how our model looks after hyper-parameter tuning 
        print(grid.best_estimator_) 

        break


def main():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    #print(df_train["Name"].apply(getTitleFromName).value_counts())
    #print(df_test["Name"].apply(getTitleFromName).value_counts())

    #print(df_train["Ticket"].describe(include="all"))
    #print(df_test["Ticket"].describe(include="all"))
    #showAgeVsTitleCorr(df_train)
    #plotTickets(df_train)
    #showFareVsFare_Person(df_train)
    #print("CabinFL: {}".format(df_train["Cabin"].dropna().apply(lambda x: x[0]).value_counts()))
    #print(df_train[oriFeatures].describe(include='all'))
    #print(df_train["Embarked"].value_counts())
    #print(df_train["Cabin"].value_counts())
    #print(df_train["Pclass"].value_counts())
    #showAgeVsSurvival(df_train)
    #showAgeOntologyVsSurvival(df_train)
    #showAgeCAOVsSurvival(df_train)
    #showFamilyCountVsSurvival(df_train)
    #showInvAgeVsSurvival(df_train)
    #showLogAgeVsSurvival(df_train)
    #showFareVsSurvival(df_train)
    #showLogFareVsSurvival(df_train)
    #showSqrtFareVsSurvival(df_train)
    #showSqrtAgeVsSurvival(df_train)
    #showAgeGenderVsSurvival(df_train)
    #showAgeXPclassVsSurvival(df_train)
    #showFareVsPClass(df_train)
    #showEmbarkedFareVsSurvived(df_train)
    #showSiblingSpouseVsSurvived(df_train)
    #showFamilySizeVsSurvived(df_train)
    #showFamilySize2VsSurvived(df_train)
    #showSibSpParchPclass(df_train)
    #print("Unique cabin numbers: {}".format(np.unique(df_train["Cabin"].dropna())))
    #showCabinVsSurvived(df_train)
    #showAgeQcutVsSurvival(df_train)
    #obtainBinningForFareAndAge(df_train, df_test)
    #showFareQcutVsSurvival(df_train)
    #showCabinVsFare(df_train)
    #showTitleVsSurvival(df_train)
    #showFareDist(df_train)
    rfattempt1(df_train, df_test)

    #searchParamsMaybeIllegal(df_train)
    #filterFeaturesQuestionable(df_train)
    #showTitles(df_train, df_test)

    #df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
    #               'C': [1, 2, 3]})
    #df = df.join(pd.get_dummies(df[['A', 'B']], prefix=['col1', 'col2']))
    #print(df)
    

if __name__ == '__main__':
    main()
