
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

import xgboost
from sklearn.ensemble import RandomForestClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

class FeatureSelector( TransformerMixin, BaseEstimator ):
    def __init__(self, feature_names):
        self._feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self._feature_names]
    
    def get_params(self, deep=False):
        return {"feature_names": self._feature_names}

class FeatureDropper( TransformerMixin, BaseEstimator ):
    def __init__(self, feature_names):
        self._feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.drop(self._feature_names, axis=1)
        return X
    
    def get_params(self, deep=False):
        return {"feature_names": self._feature_names}

class ModeFiller( TransformerMixin, BaseEstimator ):
    def __init__(self, feats):
        self._feats = feats
        pass

    def fit(self, X, y=None):
        self._mode_X = X[self._feats].mode().iloc[0,:]
        return self

    def transform(self, X, y=None):
        dfm = self._mode_X
        for feat,med in zip(dfm.index.values, dfm.values):
            X.loc[:, feat] = X[feat].fillna(med)
        return X
    
    def get_params(self, deep=False):
        return {"feats": self._feats}

class MapTransformer ( TransformerMixin, BaseEstimator ):
    def __init__(self, col, replace_map):
        super().__init__()
        self._replace_map = replace_map
        self._col = col

    def get_params(self, deep=False):
        return {"replace_map": self._replace_map, "col" : self._col}
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[:,self._col] = X[self._col].map(self._replace_map)
        return X

class FrequencyTransformer ( TransformerMixin, BaseEstimator ):
    def __init__(self, feats):
        self._feats = feats

    def get_params(self, deep=False):
        return {"feats": self._feats}

    def fit(self, X, y=None):
        count_map = {}
        for col in self._feats:
            count_map[col] = X[col].value_counts()
        
        self._count_map = count_map
        return self

    def transform(self, X, y=None):
        count_map = self._count_map
        for col in self._feats:
            ba = count_map[col]
            X.loc[:, col] = X[col].map(ba)
        return X

class MedianPerClassFiller( TransformerMixin, BaseEstimator ):
    def __init__(self, group_by_col, target_col):
        self._by = group_by_col
        self._on = target_col
    
    def fit(self, X, y=None):
        by = self._by
        on = self._on

        if by is None:
            self._median = X[on].median()

        else:
            # Compute the number of values and the mean of each group
            agg = X.groupby(by)[on].agg(['median'])
            md = agg["median"]
            mp = {}

            for k,v in zip(md.index.values, md.values):
                mp[k] = v
            
            self._map = mp

        return self

    def transform(self, X, y=None):
        by = self._by
        on = self._on

        if by is None:
            X.loc[:, on] = X[on].fillna(self._median)

        else:
            mp = self._map
            def fn(row):
                # NaN will be true here
                if row[on] != row[on]:
                    return mp[row[by]]
                else:
                    return row[on]

            X.loc[:, on] = X.apply(fn, axis=1)

        return X

    def get_params(self, deep=False):
        return {'group_by_col' : self._by, 'target_col' : self._on}

class FamilyTransformer( TransformerMixin, BaseEstimator ):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        if y is None:
            print("[WARN] FamilyTransformer y is None. Cannot work if Survived column not given!")
            return self

        if "Survived" in X.columns.values:
            Xy = X
        else:
            Xy = X.join(y)

        survmap = {}
        def fn(row):
            if row.Survived == row.Survived:
                # Pclass, FamilyName, FamilySize
                key = (row.Pclass, row.FamilyName, row.FamilySize)
                if key in survmap:
                    survmap[key][0] += row.Survived
                    survmap[key][1] += 1

                else:
                    # (survived, |fam|, |male surv|, |fem surv|)
                    survmap[key] = [row.Survived, 1, 0, 0]
                    
                if row.Sex == 'female':
                    survmap[key][3] += row.Survived
                else:
                    survmap[key][2] += row.Survived

            return np.nan

        Xy.apply(fn, axis=1)
        
        self._survmap = survmap
        return self

    def transform(self, X, y=None):
        survmap = self._survmap

        def fn(row, idx):
            key = (row.Pclass, row.FamilyName, row.FamilySize)
            if key in survmap:
                val = survmap[key]
                return val[idx] / val[1]
            else:
                return -1

        X.loc[:, "MaleSurv"] = X.apply(lambda row: fn(row, 2), axis=1)
        X.loc[:, "FemaleSurv"] = X.apply(lambda row: fn(row, 3), axis=1)
        return X

class OldestSurvTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, title):
        self._title = title

    def fit(self, X, y=None):
        if y is None:
            print("[WARN] FamilyTransformer y is None. Cannot work if Survived column not given!")
            return self

        if "Survived" in X.columns.values:
            Xy = X
        else:
            Xy = X.join(y)
        
        survmap = {}
        def fn(row, title):
            if row.Survived == row.Survived and row.Title == title:
                # Pclass, FamilyName, FamilySize
                key = (row.Pclass, row.FamilyName, row.FamilySize)
                if key in survmap:
                    old = survmap[key]
                    if old[0] < row.Age:
                        survmap[key] = (row.Age, row.Survived)
                else:
                    survmap[key] = (row.Age, row.Survived)
        
        Xy.apply(lambda row: fn(row, self._title), axis=1)
        self._survmap = survmap

        return self

    def transform(self, X, y=None):
        survmap = self._survmap
        def transform(row):
            key = (row.Pclass, row.FamilyName, row.FamilySize)
            if key in survmap:
                return survmap[key][1]
            else:
                return -1
        
        X.loc[:, "OldestSurv"] = X.apply(transform, axis=1)
        print(X["OldestSurv"].value_counts())
        return X

    def get_params(self, deep=False):
        return {"title" : self._title}
    
def isnan(x):
    return x != x

class NullOrNotTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, feature_names):
        self._feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for col in self._feature_names:
            X.loc[:, f"NN_{col}"] = X[col].apply(lambda x : 0 if isnan(x) else 1)
        return X
    
    def get_params(self, deep=False):
        return {"feature_names": self._feature_names}


# significant titles. Other titles have really low number of samples
titleMapBigOnly = {
    "mr": 'mr',
    "miss": 'miss',
    "mrs": 'mrs',
    "master": 'master'
}

def getTitleFromName(strName):
    givenName = strName.split(",")[1]
    title = givenName.split(".")[0].lower().strip()
    if title in titleMapBigOnly:
        return title
    else:
        return 'other'

def getFamilyName(strName):
    familyName = strName.split(',')[0]
    return familyName.lower()

def analyze(df_all):
    survmap = {}
    def fn(row, title):
        if row.Survived == row.Survived and row.Title == title:
            # Pclass, FamilyName, FamilySize
            key = (row.Pclass, row.FamilyName, row.FamilySize)
            if key in survmap:
                old = survmap[key]
                if old[0] < row.Age:
                    survmap[key] = (row.Age, row.Survived)
            else:
                survmap[key] = (row.Age, row.Survived)

        return np.nan

    def transform(row):
        if row.Survived == row.Survived:
            key = (row.Pclass, row.FamilyName, row.FamilySize)
            if key in survmap:
                return survmap[key][1]
        return -1

    df_all.apply(lambda row: fn(row, 'mrs'), axis=1)
    df_all.loc[:, "OldestFemSurv"] = df_all.apply(transform, axis=1)

    df_train = df_all.iloc[:891, :]
    sns.countplot(x="OldestFemSurv", hue="Survived", data=df_train)
    plt.legend()
    plt.show()

    pass

def engineer(df_all):
    # create title feature
    df_all.loc[:, "Title"] = df_all["Name"].apply(getTitleFromName)
    df_all.loc[:, "FamilyName"] = df_all["Name"].apply(getFamilyName)

    # create #relatives feature
    df_all.loc[:, "FamilySize"] = df_all["SibSp"] + df_all["Parch"] + 1

    ticket_counts = df_all["Ticket"].value_counts()
    df_all.loc[:, "SameTickets"] = df_all["Ticket"].map(ticket_counts)

    #ft = FamilyTransformer()
    #ft.fit(df_all, df_all["Survived"])
    #ft.transform(df_all, df_all["Survived"])
    #print(df_all.columns.values)

def predict(df_train, df_test):
    pipeline = Pipeline(steps=[
        ('age_filler', MedianPerClassFiller("Title", "Age")),
        #('embarked_filler', ModeFiller(["Embarked"])),
        ('fare_filler', MedianPerClassFiller("Pclass", "Fare")),
        #('family_transformer', FamilyTransformer()),
        #('oldest_surv_transformer', OldestSurvTransformer('mrs')),
        ('title_transformer', FrequencyTransformer(["Title"])),
        ('sex_map_transformer', MapTransformer("Sex", {"female": 0, 'male': 1})),
        #('cabin_transformer', NullOrNotTransformer(["Cabin"])),
        #('embarked_transformer', MapTransformer("Embarked", {"C": 0, 'Q': 1, 'S': 2})),
        ("drop_useless_features", FeatureDropper(['Pclass', 'SibSp', 'Parch', "FamilyName", 'Cabin'])),
        ('model', xgboost.XGBClassifier())
        #('model', RandomForestClassifier())
    ])

    train_col = ["Pclass", 'Title', "FamilyName", "FamilySize", 'Sex', 'Age', 
    'SibSp', 'Parch', 'SameTickets' , 'Fare', 'Cabin']
    print(np.mean(cross_val_score(pipeline, df_train[train_col], df_train["Survived"], cv=5)))

    pipeline.fit(df_train[train_col], df_train["Survived"])

    model = pipeline.steps[5][1]
    print(model.feature_importances_)
    #print(model.get_feature_names())
    


    train_pred = pipeline.predict(df_train[train_col]).astype(int)
    df_train.loc[:, "TrainPred"] = train_pred
    df_train.to_csv('train.evaluation.pxgb.csv', index=False)

    test_pred = pipeline.predict(df_test[train_col]).astype(int)
    df_test.loc[:, "Survived"] = test_pred
    df_test[["PassengerId", "Survived"]].to_csv("test.predicted.pxgb.csv", index=False)

def main():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    df_all = df_train.append(df_test)
    engineer(df_all)
    print("Engineer done")

    #analyze(df_all)

    df_train = df_all.iloc[:891, :]
    df_test = df_all.iloc[891:, :]

    print("Start training")
    predict(df_train, df_test)

if __name__ == '__main__':
    main()
