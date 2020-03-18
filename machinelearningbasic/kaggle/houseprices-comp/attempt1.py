
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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

def smoothMeanLabelling(X_train, y_train, X_test, featureColumn, targetColumn="SalePrice", m=None):
    Xy_train = X_train.join(y_train)
    wm = Xy_train.shape[0]
    if m is not None:
        wm = m
    smoothed = calc_smooth_mean(Xy_train, by=featureColumn, on=targetColumn, m=wm)
    #print(f"{featureColumn} -> {targetColumn}: {smoothed}")
    X_train.loc[:,featureColumn] = X_train[featureColumn].map(smoothed)
    X_test.loc[:,featureColumn] = X_test[featureColumn].map(smoothed)
    return X_train, X_test

def fillNaN(X_train, X_test, col, value):
    X_train.loc[:,col] = X_train[col].fillna(value)
    X_test.loc[:,col] = X_test[col].fillna(value)
    return X_train, X_test

def applyTransform(X_train, X_test, col, fnTransform):
    X_train.loc[:,col] = X_train[col].apply(fnTransform)
    X_test.loc[:,col] = X_test[col].apply(fnTransform)
    return X_train, X_test

def applyOHE(X_train, X_test, col):
    X_train = X_train.join(pd.get_dummies(X_train[col], prefix=col))
    X_train = X_train.drop([col], axis=1)

    X_test = X_test.join(pd.get_dummies(X_test[col], prefix=col))
    X_test = X_test.drop([col], axis=1)

    return X_train, X_test

def trainML(X_train, y_train, X_test, displayImportances=False):
    model = XGBRegressor(objective='reg:squarederror')
    model = model.fit(X_train, y_train)
    
    if displayImportances:
        importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model.feature_importances_,3)})
        importances = importances.sort_values('importance',ascending=False).set_index('feature')
        for f,v in zip(importances.index.values, importances.values):
            print(f"{f}\t{v}")

    y_pred = model.predict(X_test)
    return model, y_pred

def trainLinearRegression(X_train, y_train, X_test):
    model = LinearRegression()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def stdScale(X_train, X_test, columnName):
    fsStdScaler = StandardScaler()
    fsStdScaler = fsStdScaler.fit(X_train[[columnName]])
    X_train.loc[:,columnName] = fsStdScaler.transform(X_train[[columnName]])
    X_test.loc[:,columnName] = fsStdScaler.transform(X_test[[columnName]])
    return X_train, X_test

def replaceNAInData(df):
    """
    NA is interpreted as Pandas as N/A or null. However, in this context
    some features interpreted NA differently. 
    """
    df.loc[:,"Alley"] = df["Alley"].fillna("NoAlleyAccess")
    df.loc[:,"BsmtQual"] = df["BsmtQual"].fillna("NoBasement")
    df.loc[:,"BsmtCond"] = df["BsmtCond"].fillna("NoBasement")
    df.loc[:,"BsmtExposure"] = df["BsmtExposure"].fillna("NoBasement")
    df.loc[:,"BsmtFinType1"] = df["BsmtFinType1"].fillna("NoBasement")
    df.loc[:,"BsmtFinType2"] = df["BsmtFinType2"].fillna("NoBasement")
    df.loc[:,"FireplaceQu"] = df["FireplaceQu"].fillna("NoFireplace")
    df.loc[:,"GarageType"] = df["GarageType"].fillna("NoGarage")
    df.loc[:,"GarageFinish"] = df["GarageFinish"].fillna("NoGarage")
    df.loc[:,"GarageQual"] = df["GarageQual"].fillna("NoGarage")
    df.loc[:,"GarageCond"] = df["GarageCond"].fillna("NoGarage")
    df.loc[:,"PoolQC"] = df["PoolQC"].fillna("NoPool")
    df.loc[:,"Fence"] = df["Fence"].fillna("NoFence")
    df.loc[:,"MiscFeature"] = df["MiscFeature"].fillna("NoMiscFeature")
    return df

def findNaNRows(df):
    return df[df.isna().any(axis=1)]

def findNaNColumns(df):
    return df.columns[df.isna().any()].tolist()

def transform_train_test(X_train, X_test, y_train, y_test=None):

    targetCol = "SalePrice"

    thrownOutCols = [
        "Utilities", "MoSold", "YrSold", "Functional",
        "MiscFeature", "Heating", "MiscVal", "PoolQC", "Condition2",
        "YearBuilt", "YearRemodAdd", "GarageYrBlt", "BsmtHalfBath", "Street",
        "PoolArea", "BldgType", "BsmtFinSF2", "LowQualFinSF", "BsmtFullBath"
    ]

    def ino(row, col, l, dv, prefix = ''):
        if row[col] in l:
            return prefix + str(row[col])
        else:
            return dv

    def NeighborhoodTransformer(row):
        if row["Neighborhood"] == 'NPkVill' or row["Neighborhood"] == 'Blueste':
            return "Other"
        else:
            return row["Neighborhood"]

    def SaleTypeTransformer(row):
        return ino(row, "SaleType", ["WD", "New", "COD"], "Oth")

    def MSSubClassTransformer(row):
        return ino(row, "MSSubClass", [20,60,50,120,30,160,70,80,90,190,85], "Other", 'MSS')

    def RoofStyleTransformer(row):
        return ino(row, "RoofStyle", ["Gable", "Hip"], "Other")

    def Exterior1stTransformer(row):
        return ino(row, "Exterior1st", ["VinylSd","HdBoard","MetalSd","Wd Sdng","Plywood","CemntBd","BrkFace"], "Other")

    def Exterior2ndTransformer(row):
        return ino(row, "Exterior2nd", ["VinylSd","MetalSd","HdBoard","Wd Sdng","Plywood","CmentBd","Wd Shng"], "Other")

    def SaleConditionTransformer(row):
        return ino(row, "SaleCondition", ["Normal","Partial","Abnorml","Family"], "Other")

    # don't do anything to these cols
    syntheticCols = [
        ("HasMasonVeneer", lambda row: 1 if row.MasVnrType != "None" else 0), 
        ("HasBasement", lambda row: 1 if row.BsmtQual != "NoBasement" else 0), 
        ("Has2ndFloor", lambda row: 1 if row["2ndFlrSF"] > 0.0 else 0), 
        ("HasGarage", lambda row: 1 if row.GarageType != "NoGarage" else 0), 
        ("HasWoodDeck", lambda row: 1 if row.WoodDeckSF > 0.0 else 0), 
        ("HasOpenPorch", lambda row: 1 if row.OpenPorchSF > 0.0 else 0), 
        ("HasEnclosedPorch", lambda row: 1 if row.EnclosedPorch > 0.0 else 0), 
        ("Has3SsnPorch", lambda row: 1 if row["3SsnPorch"] > 0.0 else 0), 
        ("HasScreenPorch", lambda row: 1 if row["ScreenPorch"] > 0.0 else 0), 
        ("HasPool", lambda row: 1 if row.PoolQC != "NoPool" else 0), 
        ("HasFence", lambda row: 1 if row.Fence != "NoFence" else 0), 
        ("Electrical", lambda row: 1 if row.Electrical == 'SBrkr' else 0),
        ("RoofMatl", lambda row: 1 if row.RoofMatl == 'CompShg' else 0),
        ("IsTypFunctional", lambda row: 1 if row.Functional == 'Typ' else 0),
        ("Exterior2nd", Exterior2ndTransformer),
        ("Exterior1st", Exterior1stTransformer),
        ("IsHeatingGasA", lambda row: 1 if row.Heating == 'GasA' else 0),
        ("RoofStyle", RoofStyleTransformer),
        ("MSSubClass", MSSubClassTransformer),
        ("OverallQual", lambda row: row.OverallQual if row.OverallQual > 2 else 2),
        ("Neighborhood", NeighborhoodTransformer),
        ("SaleType", SaleTypeTransformer),
        ("IsOverallCondAvgOrAbove", lambda row: 1 if row.OverallCond >= 5 else 0),
        ("YearBuildP20", lambda row: row.YearBuilt // 20),
        ("YearRemodAddP20", lambda row: row.YearRemodAdd // 20),
        ("GarageYrBltP20", lambda row: min(100, max(95, row.GarageYrBlt // 20))),
        ("SaleCondition", SaleConditionTransformer), 
        ("IsRemodelled", lambda row: 0 if row.YearBuilt == row.YearRemodAdd else 1),
        ("HouseAge", lambda row: row.YrSold - row.YearRemodAdd)
    ]

    # just use stdScaler
    asisCols = [
        "OverallCond", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", 
        "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "GarageCars"
    ]

    labelledOrdinalCols = [
        ("Fireplaces", {0: "Fp0", 1: "Fp1", 2: "Fp2", 3: "Fp2", 4: "Fp2"}),
        ("LotShape", {"Reg": 1, "IR1": 2, "IR2": 2, "IR3": 2}),
        ("LandSlope", {"Gtl": 1, "Mod": 2, "Sev": 2}),
        ("ExterQual", {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}),

        # combine good+ex=above Avg, Avg, and Below Avg class.
        ("ExterCond", {"Ex": 5, "Gd": 5, "TA": 3, "Fa": 1, "Po": 1}),
        ("BsmtQual", {"Ex": 6, "Gd": 5, "TA": 4, "Fa": 3, "Po": 2, "NoBasement": 1}),

        # 6 doesn't exist, too little Po.
        ("BsmtCond", {"Ex": 5, "Gd": 5, "TA": 4, "Fa": 3, "Po": 3, "NoBasement": 1}),
        ("BsmtExposure", {"Gd": 5, "Av": 4, "Mn": 3, "No": 2, "NoBasement": 1}),
        ("BsmtFinType1", {"GLQ": 7, "ALQ": 6, "BLQ": 5, "Rec": 4, "LwQ": 3, "Unf": 2, "NoBasement": 1}),
        ("BsmtFinType2", {"GLQ": 7, "ALQ": 6, "BLQ": 5, "Rec": 4, "LwQ": 3, "Unf": 2, "NoBasement": 1}),

        # only 1 Po instance, combine with Fa.
        ("HeatingQC", {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 2}),
        ("CentralAir", {"N": 0, "Y": 1}),
        ("KitchenQual", {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}),
        ("FireplaceQu", {"Ex": 6, "Gd": 5, "TA": 4, "Fa": 3, "Po": 2, "NoFireplace": 1}),
        ("GarageFinish", {"Fin": 4, "RFn": 3, "Unf": 2, "NoGarage": 1}),

        # too little Ex and Po.
        ("GarageQual", {"Ex": 5, "Gd": 5, "TA": 4, "Fa": 3, "Po": 3, "NoGarage": 1}),
        ("GarageCond", {"Ex": 5, "Gd": 5, "TA": 4, "Fa": 3, "Po": 3, "NoGarage": 1}),

        ("PavedDrive", {"Y": 3, "P": 2, "N": 1}),
        ("Condition1", {"Norm":"N", "Feedr": "F", "Artery": "A", "RRAn": "R", "RRAe": "R", "RRNn": "R", "RRNe": "R", "PosN": "P", "PosA": "P"}),
        ("Foundation", {"BrkTil": "BT", "CBlock": "CB", "PConc": "PC", "Slab": "OT", "Stone": "OT", "Wood": "OT"}),
        ("LotConfig", {"Inside": "Inside", "Corner": "Corner", "CulDSac": "CulDSac", "FR2": "FR23", "FR3": "FR23"}),
        ("GarageType", {"Attchd": "Attchd", "Detchd": "Detchd", "BuiltIn": "BuiltIn", "NoGarage": "NoGarage", "Basment": "Others", "CarPort": "Others", "2Types": "Others" })
    ]

    # can be processed with skew LR
    continuousCols = [
        "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
        "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
        "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", 
        
        # these will be removed later. It is required for other feature
        "YearBuilt", "YearRemodAdd", "GarageYrBlt"
    ]

    smoothedMeanCols = [
        "MSSubClass", "MSZoning", "LotConfig", "Neighborhood", 
        "Condition1", "BldgType", "HouseStyle",
        "RoofStyle", "Exterior1st", "Exterior2nd",
        "MasVnrType", "Foundation", "GarageType", "SaleType", "SaleCondition",
        "Alley", "LandContour", "Fireplaces", "Fence", 
        "Electrical", "RoofMatl", "HasMasonVeneer", "LotShape", "LandSlope",
        "YearBuildP20", "YearRemodAddP20", "GarageYrBltP20"
    ]

    oheCols = [
        
    ]

    # fill actual NaN values with mode and medians
    for cols in [oheCols, smoothedMeanCols]:
        for col in cols:
            if col in X_train.columns:
                tmp = X_train[col].mode().values[0]
                fillNaN(X_train, X_test, col, tmp)
    
    for col,_ in labelledOrdinalCols:
        if col in X_train.columns:
            tmp = X_train[col].mode().values[0]
            fillNaN(X_train, X_test, col, tmp)
    
    for cols in [continuousCols, asisCols]:
        for col in cols:
            if col in X_train.columns:
                tmp = X_train[col].median()
                fillNaN(X_train, X_test, col, tmp)

    for col, fn in syntheticCols:
        X_train.loc[:, col] = X_train.apply(fn, axis=1)
        X_test.loc[:, col] = X_test.apply(fn, axis=1)
    
    # transform the column values
    for col,mapping in labelledOrdinalCols:
        X_train.loc[:, col] = X_train[col].map(mapping)
        X_test.loc[:, col] = X_test[col].map(mapping)

    for col in oheCols:
        X_train, X_test = applyOHE(X_train, X_test, col)

    for col in continuousCols:
        skewTrain = X_train[col].skew()
        #skewTest = X_test[col].skew()
        #skewIsPerformed = True

        if (skewTrain >= 10.0):
            # do log(x+1)
            X_train, X_test = applyTransform(X_train, X_test, col, lambda x : math.log(x+1))

        elif (skewTrain >= 1.0):
            # do sqrt(x)
            X_train, X_test = applyTransform(X_train, X_test, col, lambda x : math.sqrt(x))

        else:
            #skewIsPerformed = False            
            pass

        #if skewIsPerformed:
        #    skewTrainAfter = X_train[col].skew()
        #    skewTestAfter = X_test[col].skew()

            #if abs(skewTrainAfter) >= 1.0 or abs(skewTestAfter) >= 1.0:
            #    print(f"[warn] Distribution is still HIGHLY skewed for {col}, " + 
            #          f"Before: {skewTrain}, {skewTest}, After {skewTrainAfter}, {skewTestAfter}")

    def sp_transform(x):
        return math.log(x+1)
    
    def sp_inv_transform(x):
        return math.exp(x) - 1

    # transform the SalePrice
    y_train = y_train.apply(sp_transform)
    
    # scale the SalePrice using StandardScaler
    Xy_train = X_train.join(y_train)
    stdScalerModel = StandardScaler()
    #print("Xy_train[[targetCol]]: {}".format(Xy_train[targetCol][0:5]))
    stdScalerModel = stdScalerModel.fit(Xy_train[[targetCol]])
    Xy_train.loc[:,f"{targetCol}Std"] = stdScalerModel.transform(Xy_train[[targetCol]])
    y_train_stdscaled = Xy_train[f"{targetCol}Std"]

    for col in smoothedMeanCols:
        X_train, X_test = smoothMeanLabelling(X_train, y_train_stdscaled, X_test, col, f"{targetCol}Std", 300)

    for col in continuousCols:
        X_train, X_test = stdScale(X_train, X_test, col)

    X_train = X_train.drop(thrownOutCols, axis=1)
    X_test = X_test.drop(thrownOutCols, axis=1)

    # return mean_squared_error
    #print("y_train: {}".format(y_train[0:5]))

    def printNans(df, title):
        nanCols = findNaNColumns(df)
        if len(nanCols) > 0:
            print("ERROR {} Columns With NaN values!!! {}".format(title, nanCols))

    printNans(X_train, "train")
    printNans(X_test, "test")

    #print("X_train shape: {}".format(X_train.shape))
    #print("X_test shape: {}".format(X_test.shape))
    _, y_pred = trainML(X_train, y_train_stdscaled, X_test, y_test is None)

    #print("y_pred length: {}".format(len(y_pred)))
    X_test.loc[:,f"{targetCol}Pred"] = y_pred
    X_test.loc[:,f"{targetCol}Pred"] = stdScalerModel.inverse_transform(X_test[[f"{targetCol}Pred"]])

    # remember that previously, before the stdScaler, the targetCol is transformed
    # using sqrt
    y_pred_rtf = [sp_inv_transform(x) for x in X_test[f"{targetCol}Pred"]]

    print("Done train test")
    if y_test is not None:
        #print("y_test: {}".format(y_test[0:5]))
        #print("y_pred_rtf: {}".format(y_pred_rtf[0:5]))
        return mean_absolute_error(y_test, y_pred_rtf)
    else:
        return y_pred_rtf

def kfold_train_test(df_train, df_forSubs):
    df_train = replaceNAInData(df_train)
    df_forSubs = replaceNAInData(df_forSubs)

    kf = KFold(n_splits=5)

    oriFeatures = df_train.columns.drop(["Id", "SalePrice"])
    X = df_train[oriFeatures].copy()
    y = df_train["SalePrice"].copy()

    meansqerr = []

    for train_idx, test_idx in kf.split(X):
        print("Begin splitting data to train/test pair")

        # read the following to know why .copy() is required!
        # https://www.dataquest.io/blog/settingwithcopywarning/
        X_train, X_test = X.iloc[train_idx,:].copy(), X.iloc[test_idx,:].copy()
        y_train, y_test = y[train_idx].copy(), y[test_idx].copy()

        print("Start transforming, test, train")

        err = transform_train_test(X_train, X_test, y_train, y_test)

        meansqerr.append(err)
        
    print("Avg Mean Squared Error: {}".format(np.mean(meansqerr)))

    print("Predicting result in test.csv")
    X_p = df_forSubs[oriFeatures].copy()

    y_p = transform_train_test(X, X_p, y)

    df_forSubs.loc[:,"SalePrice"] = y_p
    df_forSubs[["Id", "SalePrice"]].to_csv("test.predicted.attempt1.csv", index=False)

    print("Done")

if __name__ == '__main__':
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    kfold_train_test(df_train, df_test)

