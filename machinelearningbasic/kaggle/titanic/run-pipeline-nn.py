
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_auc_score

import tensorflow as tf

class OHETransformer( TransformerMixin, BaseEstimator ):
    def __init__(self, feats):
        super().__init__()
        self._feats = feats
    
    def fit(self, X, y=None):
        self._ohe = OneHotEncoder(handle_unknown='error', sparse=False, drop='first')
        self._ohe.fit(X[self._feats])
        return self
    
    def transform(self, X, y=None):
        tX = self._ohe.transform(X[self._feats])
        feat_names = self._ohe.get_feature_names(self._feats)
        X = X.drop(self._feats, axis=1)
        dftX = pd.DataFrame(data=tX, columns=feat_names)
        X = X.join(dftX)
        return X

class StandardScalerTransformer( TransformerMixin, BaseEstimator ):
    def __init__(self, feats):
        super().__init__()
        self._feats = feats
    
    def fit(self, X, y=None):
        self._scaler = StandardScaler()
        self._scaler.fit(X[self._feats])
        return self

    def transform(self, X, y=None):
        tX = self._scaler.transform(X[self._feats])
        X.loc[:, self._feats] = tX
        return X
    
    def get_params(self, deep=False):
        return {"feats": self._feats}

class ScalerTransformer( TransformerMixin, BaseEstimator ):
    def __init__(self, feats, fn):
        self._feats = feats
        self._fn = fn

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for col in self._feats:
            X.loc[:, col] = X[col].apply(self._fn)
        return X
    
    def get_params(self, deep=False):
        return {"feats": self._feats, "fn": self._fn}

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

def getTitleFromName(strName):
    givenName = strName.split(",")[1]
    title = givenName.split(".")[0].lower().strip()
    if title == 'mr':
        return 'mr'
    elif title == 'miss':
        return 'miss'
    elif title == 'mrs':
        return 'mrs'
    elif title == 'master':
        return 'master'
    else:
        return 'other'

def preprocess_df(df_all):
    df_all.loc[:, "Title"] = df_all["Name"].apply(getTitleFromName)
    df_all.loc[:, "FamilySize"] = df_all["SibSp"] + df_all["Parch"] + 1
    return df_all

denselayers = [[8], [10], [8, 8], [10, 5], [12, 6, 4]]
def build_nn(input_cols = 7, opt = 'Adam', dropoutrate = 0.2, lyrsi = 2):

    layers = []
    layers.append(tf.keras.layers.Input(shape=(input_cols,)))
    for units in denselayers[lyrsi]:
        layers.append(tf.keras.layers.Dense(units=units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        layers.append(tf.keras.layers.Dropout(rate=dropoutrate))
    layers.append(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    model = tf.keras.Sequential(layers=layers)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

    return model

def create_pipeline(model):
    # feature_pipeline does not have the capacity to append from unaltered features
    # they have to be added manually.
    #noop_pipeline = Pipeline(steps=[
    #    ('cat_selector', FeatureSelector(['Sex']))
    #])

    #numerical_pipeline = Pipeline(steps=[
    #    ('cat_selector', FeatureSelector(['Age', 'FamilySize', 'Fare'])),

        # https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
        # good rule of thumb is the data to be scaled to [0,1], or mean = 0, stdev = 1
        # standardization is good if the distribution is closing to normal dist.
        # otherwise, use normalization
    #    ('std_scaler', StandardScaler())
    #])

    #categorical_pipeline = Pipeline(steps=[
    #    ('cat_selector', FeatureSelector(["Pclass", 'Title', 'Embarked'])),
    #    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=True))
    #])

    # Keras with validation_split seems not supporting FeatureUnion, unfortunately.
    # because FeatureUnion outputs the result not in np array or tensors, ofc.
    # but some sparse matrix or something.
    #feature_pipeline = FeatureUnion(transformer_list=[
    #    ('categorical', categorical_pipeline),
    #    ('numerical', numerical_pipeline),
    #    ('nop', noop_pipeline)
    #])

    pipeline = Pipeline(steps=[
        ('age_filler', MedianPerClassFiller("Title", "Age")),
        ('embarked_filler', ModeFiller(["Embarked"])),
        ('fare_filler', MedianPerClassFiller("Pclass", "Fare")),

        # https://visualstudiomagazine.com/articles/2014/01/01/how-to-standardize-data-for-neural-networks.aspx
        # it seems that for binary data, it is better to encode to -1,1 rather than 0,1., but not here it seems.
        ('sex_map_transformer', MapTransformer("Sex", {"female": -1, 'male': 1})),
        ('pclass_map_transformer', MapTransformer("Pclass", {1: "High", 2: "Middle", 3: "Low"})),
        #('title_transformer', FrequencyTransformer(["Title"])),
        #('embarked_transformer', FrequencyTransformer(["Embarked"])),
        #('pclass_transformer', FrequencyTransformer(["Pclass"])),
        #('age_log', ScalerTransformer(["Age"], lambda x : math.log(x+1))),
        #('fare_log', ScalerTransformer(["Fare"], lambda x : math.log(x+1))),
        ('ohe_transformer', OHETransformer(["Pclass", 'Title', 'Embarked'])),
        ('std_scaler', StandardScalerTransformer(['Age', 'FamilySize', 'Fare'])),
        #('feature_pipeline', feature_pipeline),
        ('model', model)
    ])

    return pipeline

def dump_grid_search_results(grid_result):
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def search_layers_dropoutrate(df_train):
    train_col = ["Pclass", 'Title', "FamilySize", 'Sex', 'Age', 'Fare', 'Embarked']
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_nn, opt = 'Nadam', input_cols=15, batch_size=16, epochs=100, verbose=0)
    pipeline = create_pipeline(model)

    lyrsis = [x for x in range(0, len(denselayers))]
    dropoutrate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    param_grid = dict(model__lyrsi = lyrsis, model__dropoutrate = dropoutrate)

    #param_grid = {
    #    "model__denselayer" : denselayers,
    #    "model__dropoutrate" : dropoutrate
    #}

    # search the grid
    grid = GridSearchCV(estimator=pipeline, 
                        param_grid=param_grid,
                        cv=5,
                        verbose=2, 
                        n_jobs=-1)  # include n_jobs=-1 if you are using CPU

    grid_result = grid.fit(df_train[train_col], df_train["Survived"])
    dump_grid_search_results(grid_result)


def search_opt(df_train):
    train_col = ["Pclass", 'Title', "FamilySize", 'Sex', 'Age', 'Fare', 'Embarked']
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_nn, input_cols=15, batch_size=16, epochs=100, verbose=0)
    pipeline = create_pipeline(model)

    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']

    param_grid = {
        "model__opt" : optimizers
    }

    # search the grid
    grid = GridSearchCV(estimator=pipeline, 
                        param_grid=param_grid,
                        cv=5,
                        verbose=2, 
                        n_jobs=-1)  # include n_jobs=-1 if you are using CPU

    grid_result = grid.fit(df_train[train_col], df_train["Survived"])
    dump_grid_search_results(grid_result)


def search_params(df_train):
    train_col = ["Pclass", 'Title', "FamilySize", 'Sex', 'Age', 'Fare', 'Embarked']
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_nn, input_cols=15, verbose=0)
    pipeline = create_pipeline(model)

    batch_size = [16, 32, 64]
    epochs = [25, 50, 75, 100]

    param_grid = {
        "model__batch_size": batch_size,
        "model__epochs": epochs
    }

    # search the grid
    grid = GridSearchCV(estimator=pipeline, 
                        param_grid=param_grid,
                        cv=5,
                        verbose=2, 
                        n_jobs=-1)  # include n_jobs=-1 if you are using CPU

    grid_result = grid.fit(df_train[train_col], df_train["Survived"])
    dump_grid_search_results(grid_result)


def plot_acc(train, validation, blurb):
    plt.plot(train)
    plt.plot(validation)
    plt.title(f'model {blurb}')
    plt.ylabel(blurb)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

def predict(df_train, df_test):
    train_col = ["Pclass", 'Title', "FamilySize", 'Sex', 'Age', 'Fare', 'Embarked']

    callbacks = [
        LossHistory()
    ]
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_nn, dropoutrate = 0.3, lyrsi = 0, opt = 'Nadam', input_cols = 12, epochs=100, batch_size=16, verbose=0)
    pipeline = create_pipeline(model)

    scores = cross_val_score(pipeline, df_train[train_col], df_train["Survived"], cv=5)

    print(f"mean train acc: {np.mean(scores)}")

    pipeline.fit(df_train[train_col], df_train["Survived"], model__callbacks=callbacks, model__validation_split=0.2)
    train_pred = pipeline.predict(df_train[train_col]).astype(int)

    print("roc: {}".format(roc_auc_score(df_train["Survived"], train_pred)))

    plot_acc(callbacks[0].losses, callbacks[0].val_losses, 'loss')

    test_pred = pipeline.predict(df_test[train_col]).astype(int)
    df_test.loc[:, "Survived"] = test_pred
    df_test[["PassengerId", "Survived"]].to_csv("test.predicted.nn.csv", index=False)

def main():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_all = df_train.append(df_test)
    df_all = preprocess_df(df_all)

    df_train = df_all.iloc[:891, :]
    df_test = df_all.iloc[891:, :]

    #search_params(df_train)
    #search_opt(df_train)
    #search_layers_dropoutrate(df_train)
    predict(df_train, df_test)

def scratchpad():
    df = pd.DataFrame(data = {
        "numeric1": [0, 0, 0, 0, 1, 1, 1, 1],
        "numeric2": [0, 0, 1, 1, 2, 2, 3, 3],
        "categ0": ['m', 'm', 'm', 'm', 'f', 'f', 'f', 'x'],
        "categ1": ['p', 'w', 'p', 'w', 'p', 'w', 'p', 'w'],
        "categ": [0, 0, 0, 0, 1, 1, 1, 1]
    })
    #ssj = StandardScalerTransformer(["numeric1", "numeric2"])
    #ssj = ssj.fit(df)
    #ssj.transform(df)
    #print(df)

    ohe = OHETransformer(["categ0", "categ1"])
    ohe = ohe.fit(df)
    df = ohe.transform(df)
    print(df)

if __name__ == '__main__':
    #scratchpad()
    main()
    pass
