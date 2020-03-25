
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from sklearn.svm import SVC

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

class MedianFiller( TransformerMixin, BaseEstimator ):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self._median_X = X.median()
        return self
    
    def transform(self, X, y=None):
        dfm = self._median_X
        for feat,med in zip(dfm.index.values, dfm.values):
            X.loc[:, feat] = X[feat].fillna(med)
        return X

class ModeFiller( TransformerMixin, BaseEstimator ):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self._mode_X = X.mode().iloc[0,:]
        return self

    def transform(self, X, y=None):
        dfm = self._mode_X
        for feat,med in zip(dfm.index.values, dfm.values):
            X.loc[:, feat] = X[feat].fillna(med)
        return X

class LogScaler( TransformerMixin, BaseEstimator ):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in X.columns.values:
            X.loc[:,col] = X[col].apply(lambda x : math.log(x+1))
        return X

class AdditionTransformer( TransformerMixin, BaseEstimator ):
    def __init__(self, abcf):
        self._abcf = abcf

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        f = self._abcf
        fnewcol = f"{f[0]}+{f[1]}"
        X.loc[:,fnewcol] = X[f[0]] + X[f[1]]
        return X
    
    def get_params(self, deep=False):
        return {"abcf": self._abcf}

def attempt(df_train, df_test):
    #df_train.loc[:, "FamilySize"] = df_train["SibSp"] + df_train["Parch"]
    #df_test.loc[:, "FamilySize"] = df_test["SibSp"] + df_test["Parch"]

    agf_pipeline = Pipeline(steps=[
        ('cat_selector', FeatureSelector(["Age", "Fare"])),
        ('median_filler', MedianFiller()),
        ('log_scaler', LogScaler())
    ])
    cat_pipeline = Pipeline(steps=[
        ('cat_selector', FeatureSelector(["Sex", "Embarked"])),
        ('mode_filler', ModeFiller()),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    fmz_pipeline = Pipeline(steps=[
        ('cat_selector', FeatureSelector(["SibSp", "Parch"])),
        ('additional', AdditionTransformer(["SibSp", "Parch"])),
        ('std_scaler', StandardScaler())
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ('pf1', agf_pipeline),
        ('pf2', cat_pipeline),
        ('pf3', fmz_pipeline)
    ])

    full_pipeline_m = Pipeline(steps=[
        ('pipeline', full_pipeline),
        ('model', SVC())
    ])

    print(cross_val_score(full_pipeline_m, df_train[["Age", "Fare", "Sex", "Embarked", "SibSp", "Parch"]], df_train["Survived"], cv=5))
    

def main():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    attempt(df_train, df_test)

if __name__ == '__main__':
    main()
