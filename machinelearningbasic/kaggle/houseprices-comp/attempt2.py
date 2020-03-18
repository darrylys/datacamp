
# Not going to attempt CV here

import sys

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

OUTPREFIX = sys.argv[0]

def findNaNColumns(df):
    return df.columns[df.isna().any()].tolist()

def printNans(df, title):
    nanCols = findNaNColumns(df)
    if len(nanCols) > 0:
        print("ERROR {} Columns With NaN values!!! {}".format(title, nanCols))

def findNaNRows(df):
    return df[df.isna().any(axis=1)]

def scatterplotshow(df, x, y, title=""):
    print(f"Corr {x} {y} is {df[x].corr(df[y], method='pearson')}")
    fig = plt.figure(figsize=(20, 6))
    plt.tight_layout()
    sns.scatterplot(x=x,y=y,data=df)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{OUTPREFIX}_s{title}.png')
    plt.close(fig)
def heatmap(df, title=''):
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(df, annot=True)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{OUTPREFIX}_h{title}.png')
    plt.close(fig)
def boxplotPlot(df, x, y, title=""):
    print(df[x].dropna().value_counts())
    fig = plt.figure(figsize=(20, 6))
    plt.tight_layout()
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{OUTPREFIX}_b{title}.png')
    plt.close(fig)
def violinPlot(df, x, y, title=""):
    print(df[x].dropna().value_counts())
    fig = plt.figure(figsize=(20, 6))
    plt.tight_layout()
    sns.violinplot(x=x,y=y,data=df)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{OUTPREFIX}_v{title}.png')
    plt.close(fig)

def fill_na_with_mode(df_train, df_submission, feats):
    for f in feats:
        tmp = df_train[f].mode().values[0]
        df_train.loc[:, f] = df_train[f].fillna(tmp)
        df_submission.loc[:, f] = df_submission[f].fillna(tmp)
def find_high_numeric_corr_with_sale_price(df, threshold=0.5):
    corr = df.corr()
    corrToSalePrice = corr["SalePrice"]

    corrToSalePriceHigh = []
    for k,v in zip(corrToSalePrice.index.values, corrToSalePrice.values):
        if abs(v) >= threshold:
            corrToSalePriceHigh.append((k, v))
    
    corrToSalePriceHigh = sorted(corrToSalePriceHigh, key=lambda x: x[1], reverse=True)
    print(f"Numeric features with correlation >= {threshold} to SalePrice:")
    for k,v in corrToSalePriceHigh:
        print(f"\t{k}\t{v}")

    heatmap(df[[x[0] for x in corrToSalePriceHigh]].corr(), "heatmap")

    # there are 10 numeric columns:
    # colinearity issue between GarageArea vs GarageCars, 
    # TotRmsAbvGrade vs GrLiveArea, and 1stAreaSF vs totalBsmtAreaSF

    return corrToSalePriceHigh
    
def find_na_values(df_train, df_test):

    def showgt0(dfs):
        knv = []
        for k,v in zip(dfs.index.values, dfs.values):
            if v > 0:
                print(f"\t{k}\t{v}")
                knv.append(k)
        return knv

    print("NA values of train df: ")
    s = showgt0(df_train.isnull().sum())

    print("\n")

    print("NA values of test df: ")
    s.extend(showgt0(df_test.isnull().sum()))

    nz = np.unique(s)
    print(f"Missing values in these cols: {nz}")

    return nz

def NA_analysis__alley(df_train, df_submission):
    # alley does not seem to be related to other features,
    # just fill NA with "None".

    print(df_train["Alley"].value_counts(dropna=False))
    print(df_submission["Alley"].value_counts(dropna=False))

    df_train.loc[:,"Alley"] = df_train["Alley"].fillna("None")
    df_submission.loc[:,"Alley"] = df_submission["Alley"].fillna("None")

def show_counts(df, cols, title):
    print(title)
    for f in cols:
        print(df[f].value_counts(dropna=False))

def NA_analysis__basement(df_train, df_submission):
    # basement features:
    #    BsmtQual    
    #    BsmtCond    
    #    BsmtExposure
    #    BsmtFinType1
    #    BsmtFinType2
    #    BsmtFinSF1  
    #    BsmtFinSF2  
    #    BsmtUnfSF   
    #    TotalBsmtSF 
    #    BsmtFullBath
    #    BsmtHalfBath

    nobsmt = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
    numerics = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

    # preliminary. print value_counts of nobsmt
    show_counts(df_train, nobsmt, 'train')
    show_counts(df_submission, nobsmt, 'test')

    # find houses with no basement:
    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2 are all NA
    def check_no_bsmt_row(row):
        # property of NaN is that NaN != NaN is true
        fs = [row[x] != row[x] for x in nobsmt]
        fsm = {True: 0, False: 0}
        for f in fs:
            fsm[f] += 1

        # if all fs == False, return 2
        if fsm[False] == len(fs):
            return 2
        
        # if all fs == True, return 0
        if fsm[True] == len(fs):
            return 0

        return 1

    def find_no_bsmt(df):
        df.loc[:, "HasBsmt"] = df.apply(check_no_bsmt_row, axis=1)
    
    find_no_bsmt(df_train)
    print(df_train["HasBsmt"].value_counts())

    find_no_bsmt(df_submission)
    print(df_submission["HasBsmt"].value_counts())

    # there are 79 entries combined, that have no basement at all.
    # check for the other 6 features, these must be 0; (HasBsmt == 0)
    def check_6_feats_for_0(df, title):
        print(f"dataframe title: {title}")
        for f in numerics:
            rdf = df[(df["HasBsmt"] == 0) & (df[f] != 0)][["Id", f]]
            if len(rdf) > 0:
                print(f"feature: {f}")
                print(rdf)
                for rid in rdf["Id"].values:
                    df.loc[df["Id"] == rid, f] = 0
    check_6_feats_for_0(df_train, 'train')
    # train has all correct No Basement 6 other features (all zeros)

    check_6_feats_for_0(df_submission, 'test')
    # there are two rows: 2121 and 2189. fill these with zeros (already performed in check_6_feats_for_0)

    for f in nobsmt:
        df_train.loc[df_train["HasBsmt"] == 0, f] = 'None'
        df_submission.loc[df_submission["HasBsmt"] == 0, f] = 'None'

    # Check 9 entries that have null values in nobsmt.
    print(df_train[df_train["HasBsmt"] == 1][["Id"] + nobsmt])
    print(df_submission[df_submission["HasBsmt"] == 1][["Id"] + nobsmt])

    # For these entries, just fill them with mode(). Since this is mode(), even if using CV,
    # since the distribution is expected to be the same, the mode won't change even if only
    # train-CV is used.
    for f in nobsmt:
        tmp = df_train[f].mode().values[0]
        df_train.loc[df_train["HasBsmt"] == 1, f] = df_train.loc[df_train["HasBsmt"] == 1, f].fillna(tmp)
        df_submission.loc[df_submission["HasBsmt"] == 1, f] = df_submission.loc[df_submission["HasBsmt"] == 1, f].fillna(tmp)

    # Check 9 entries that have null values in nobsmt, again to confirm
    print(df_train[df_train["HasBsmt"] == 1][["Id"] + nobsmt])
    print(df_submission[df_submission["HasBsmt"] == 1][["Id"] + nobsmt])

    # check NaN for HasBsmt == 1 or 2.
    def check_6_feats_for_1_or_2(df, title, fv):
        print(f"dataframe title: {title} HasBsmt: {fv}")
        for f in numerics:
            rdf = df[(df["HasBsmt"] == fv) & (df[f] != df[f])][["Id", f]]
            if len(rdf) > 0:
                print(f"feature: {f}")
                print(rdf)
    check_6_feats_for_1_or_2(df_train, 'train', 1)
    check_6_feats_for_1_or_2(df_submission, 'test', 1)
    
    check_6_feats_for_1_or_2(df_train, 'train', 2)
    check_6_feats_for_1_or_2(df_submission, 'test', 2)
    # VERY NICE, No NaN found for other bsmt related numeric variables.

    # in conclusion, only the NaN values from HasBsmt == 1 needs to be filled with statistical
    # mode test.

    show_counts(df_train, nobsmt, 'train')
    show_counts(df_submission, nobsmt, 'test')

    df_train.drop(['HasBsmt'], axis=1, inplace=True)
    df_submission.drop(['HasBsmt'], axis=1, inplace=True)

def NA_analysis__electrical(df_train, df_submission):
    # there is only one Electrical
    show_counts(df_train, ['Electrical'], 'train')
    show_counts(df_submission, ['Electrical'], 'test')

    # Electrical only has one missing value in train.
    # Vast majority is SBrkr.  Just fill with mode.
    fill_na_with_mode(df_train, df_submission, ['Electrical'])

def NA_analysis__exterior(df_train, df_submission):
    # 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond'
    feats = ['Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond']
    show_counts(df_train, feats, 'train')
    show_counts(df_submission, feats, 'test')

    # there are only 2 NaN values in Exterior1 and 2, only in test.csv
    # both are categorical, just put the mode in them.
    fill_na_with_mode(df_train, df_submission, feats[:2])

def NA_analysis__fence(df_train, df_submission):
    # fence does not seem to have any other related features
    # just replace NA with None
    df_train.loc[:,"Fence"] = df_train["Fence"].fillna("None")
    df_submission.loc[:,"Fence"] = df_submission["Fence"].fillna("None")

def NA_analysis__fireplace(df_train, df_submission):
    # there are 2 features: Fireplaces and FireplaceQu
    feats = ['Fireplaces', 'FireplaceQu']
    show_counts(df_train, feats, 'train')
    show_counts(df_submission, feats, 'test')

    # cool, for both sets, |Fireplaces == 0| == |FireplaceQu == NaN|
    # thus, we can safely replace FireplaceQu NA as None.
    df_train.loc[:,"FireplaceQu"] = df_train["FireplaceQu"].fillna("None")
    df_submission.loc[:,"FireplaceQu"] = df_submission["FireplaceQu"].fillna("None")

def NA_analysis__utilities(df_train, df_submission):
    # drop this feature, worthless
    # other than AllPub, it is NaN in test. Which means, test will contain all AllPub
    # thus, this feature cannot predict anything.
    #df_train.drop(['Utilities'], axis=1, inplace=True)
    #df_submission.drop(['Utilities'], axis=1, inplace=True)
    pass

def NA_analysis__functional(df_train, df_submission):
    show_counts(df_train, ['Functional'], 'train')
    show_counts(df_submission, ['Functional'], 'test')

    fill_na_with_mode(df_train, df_submission, ['Functional'])

def NA_analysis__kitchen(df_train, df_submission):
    feats = ['KitchenQual', 'KitchenAbvGr']

    # just 1 NA in KitchenQual. Since apparently, NA here means genuine missing data,
    # use mode to fill it.
    show_counts(df_train, feats, 'train')
    show_counts(df_submission, feats, 'test')

    fill_na_with_mode(df_train, df_submission, [feats[0]])

def NA_analysis__pool(df_train, df_submission):
    feats = ['PoolArea', 'PoolQC']

    show_counts(df_train, feats, 'train')
    show_counts(df_submission, feats, 'test')

    # VAST majority of the houses has No Pool
    # the number of houses with no pool is 10 in total (both train + test)

    # for train, all PoolArea NA can be safely replaced with None.
    df_train.loc[df_train["PoolArea"] == 0, 'PoolQC'] = df_train.loc[df_train["PoolArea"] == 0, 'PoolQC'].fillna('None')
    df_submission.loc[df_submission["PoolArea"] == 0, 'PoolQC'] = df_submission.loc[df_submission["PoolArea"] == 0, 'PoolQC'].fillna('None')

    # for test, there are 3 genuine missing data in PoolQC, with PoolArea > 0
    # It does not make sense to fill these with None (the mode), so, use Gd instead
    # which is the next mode value for PoolQC != None from train set.
    print(df_submission[df_submission['PoolQC'] != df_submission['PoolQC']][['Id'] + feats])
    # Id: 2421, 2504, 2600
    for id in [2421, 2504, 2600]:
        df_submission.loc[df_submission["Id"] == id, 'PoolQC'] = 'Gd'

def NA_analysis__mszoning(df_train, df_submission):
    show_counts(df_train, ['MSZoning'], 'train')
    show_counts(df_submission, ['MSZoning'], 'test')
    fill_na_with_mode(df_train, df_submission, ['MSZoning'])

def NA_analysis__miscfeats(df_train, df_submission):
    show_counts(df_train, ['MiscFeature'], 'train')
    show_counts(df_submission, ['MiscFeature'], 'test')

    fill_na_with_mode(df_train, df_submission, ['MiscFeature'])

def NA_analysis__sale(df_train, df_submission):
    feats = ['SaleType', 'SaleCondition']
    show_counts(df_train, feats, 'train')
    show_counts(df_submission, feats, 'test')

    fill_na_with_mode(df_train, df_submission, feats)

def isnan(x):
    return x != x

def NA_analysis__masonry(df_train, df_submission):
    feats = ['MasVnrType', 'MasVnrArea']

    show_counts(df_train, feats, 'train')
    show_counts(df_submission, feats, 'test')

    def findAreaNanTypeNan(df):
        return df[(isnan(df["MasVnrArea"])) & (isnan(df["MasVnrType"]))][["Id"] + feats]

    # check if MasVnrArea and MasVnrType is both Nan.
    # fill these with mode(MasVnrArea, MasVnrType)
    print(findAreaNanTypeNan(df_train))
    print(findAreaNanTypeNan(df_submission))
    # fill these with Area=0 & Type=None

    def fillAreaNanTypeNan(df):
        df.loc[(isnan(df["MasVnrArea"])) & (isnan(df["MasVnrType"])), feats] = ['None', 0]
    fillAreaNanTypeNan(df_train)
    fillAreaNanTypeNan(df_submission)

    def findAreaOrTypeNan(df):
        return df[(isnan(df["MasVnrArea"])) | (isnan(df["MasVnrType"]))][["Id"] + feats]
    print(findAreaOrTypeNan(df_train))
    print(findAreaOrTypeNan(df_submission))

    # fill the remaining MasVnrType with BrkFace
    df_submission.loc[df_submission["Id"] == 2611, "MasVnrType"] = 'BrkFace'

def fill_nx_with_median_per_cat(df_train, df_test, nx, cat):
    agg = df_train.groupby(cat)[nx].agg(['median'])
    medians = agg['median']
    
    medianMap = {}
    for k,v in zip(medians.index.values, medians.values):
        medianMap[k] = v
    
    def fn(row, naCol, catCol):
        if isnan(row[naCol]):
            return medianMap[row[catCol]]
        else:
            return row[naCol]

    df_train.loc[:, nx] = df_train.apply(lambda row: fn(row, nx, cat), axis=1)
    df_test.loc[:, nx] = df_test.apply(lambda row: fn(row, nx, cat), axis=1)

def NA_analysis__lot(df_train, df_submission):
    feats = ['LotFrontage', 'LotArea', 'LotShape', 'LotConfig']

    # No NA other than LotFrontage
    boxplotPlot(df_train, "Neighborhood", "LotFrontage", 'nldf-train')
    boxplotPlot(df_submission, "Neighborhood", "LotFrontage", 'nldf-test')
    # fill the LotFrontage with the median per neighborhood

    fill_nx_with_median_per_cat(df_train, df_submission, "LotFrontage", "Neighborhood")
    #return lambda df_train, df_test: fill_nx_with_median_per_cat(df_train, df_test, "LotFrontage", "Neighborhood")
    
def NA_analysis__garages(df_train, df_submission):
    hasNoGarageNA = ['GarageType','GarageFinish','GarageQual','GarageCond']
    numericGarages = ['GarageArea','GarageCars','GarageYrBlt']
    feats = hasNoGarageNA + numericGarages

    show_counts(df_train, hasNoGarageNA, 'train')
    show_counts(df_submission, hasNoGarageNA, 'test')

    def check_no_garage_row(row):
        # property of NaN is that NaN != NaN is true
        fs = [isnan(row[x]) for x in hasNoGarageNA]
        fsm = {True: 0, False: 0}
        for f in fs:
            fsm[f] += 1

        # if all fs == False, return 2
        if fsm[False] == len(fs):
            return 2
        
        # if all fs == True, return 0
        if fsm[True] == len(fs):
            return 0

        return 1
    
    df_train.loc[:,"GarageCode"]=df_train.apply(check_no_garage_row,axis=1)
    df_submission.loc[:,"GarageCode"]=df_submission.apply(check_no_garage_row,axis=1)

    def find_faulty_numeric_garage_no_garage(df):
        tmp = df[df["GarageCode"] == 0]
        return tmp[(tmp["GarageArea"] != 0) | (tmp["GarageCars"] != 0) | (tmp["GarageYrBlt"] == tmp["GarageYrBlt"])]
    print(find_faulty_numeric_garage_no_garage(df_train))
    print(find_faulty_numeric_garage_no_garage(df_submission))
    # no faulty numeric garage data. Thus, if house has no garage, 
    # GarageArea = 0, GarageCars = 0 and GarageYrBlt is NaN

    # fill GarageYrBlt <== YearBuilt
    fn = lambda row: row.YearBuilt if isnan(row.GarageYrBlt) else row.GarageYrBlt
    df_train.loc[df_train["GarageCode"] == 0,"GarageYrBlt"]=df_train[df_train["GarageCode"] == 0].apply(fn, axis=1)
    df_submission.loc[df_submission["GarageCode"] == 0,"GarageYrBlt"]=df_submission[df_submission["GarageCode"] == 0].apply(fn, axis=1)

    # if house has no garage, fill the categories as None
    for f in hasNoGarageNA:
        df_train.loc[df_train["GarageCode"] == 0,f]=df_train.loc[df_train["GarageCode"] == 0,f].fillna("None")
        df_submission.loc[df_submission["GarageCode"] == 0,f]=df_submission.loc[df_submission["GarageCode"] == 0,f].fillna("None")
    
    # if categories have no NA, check the GarageArea, GarageCars and GarageYrBlt for Nans
    def find_nan_fr(df, gc, f):
        return df[(df["GarageCode"] == gc) & (df[f] != df[f])]
    print("gc=2")
    for f in numericGarages:
        print(f"Train {f}")
        print(find_nan_fr(df_train, 2, f)[["Id"] + feats])
        print(f"Test {f}")
        print(find_nan_fr(df_submission, 2, f)[["Id"] + feats])
    # good, if categories have no NA, GarageArea, GarageCars and GarageYrBlt has no NA

    print("gc=1")
    for f in numericGarages:
        print(f"Train {f}")
        print(find_nan_fr(df_train, 1, f)[["Id"] + feats])
        print(f"Test {f}")
        print(find_nan_fr(df_submission, 1, f)[["Id"] + feats])

    # remaining, there are 2 entries with some NA
    # 2127 and 2577, in test.csv only.

    # fill GarageYrBlt <-- YearBuilt
    df_submission.loc[df_submission["GarageCode"]==1,"GarageYrBlt"]=df_submission[df_submission["GarageCode"]==1].apply(fn,axis=1)

    # the GarageFinish, GarageQual, GarageCond, use mode
    fill_na_with_mode(df_train, df_submission, ['GarageFinish', 'GarageQual', 'GarageCond'])

    # GarageArea, GarageCars is filled with median.
    def fill_median(df_train, df_test, nml):
        for nm in nml:
            tmp = df_train[nm].median()
            df_train.loc[:,nm]=df_train[nm].fillna(tmp)
            df_test.loc[:,nm]=df_test[nm].fillna(tmp)
    fill_median(df_train, df_submission, ["GarageArea", "GarageCars"])
    #return lambda df_train, df_test: fill_median(df_train, df_test, ["GarageArea", "GarageCars"])

    df_train.drop(['GarageCode'], axis=1, inplace=True)
    df_submission.drop(['GarageCode'], axis=1, inplace=True)

def PREP_Filling_NA(df_train, df_submission):
    NA_analysis__alley(df_train, df_submission)
    NA_analysis__basement(df_train, df_submission)
    NA_analysis__electrical(df_train, df_submission)
    NA_analysis__exterior(df_train, df_submission)
    NA_analysis__fence(df_train, df_submission)
    NA_analysis__fireplace(df_train, df_submission)
    NA_analysis__utilities(df_train, df_submission)
    NA_analysis__functional(df_train, df_submission)
    NA_analysis__kitchen(df_train, df_submission)
    NA_analysis__pool(df_train, df_submission)
    NA_analysis__mszoning(df_train, df_submission)
    NA_analysis__miscfeats(df_train, df_submission)
    NA_analysis__sale(df_train, df_submission)
    NA_analysis__masonry(df_train, df_submission)
    NA_analysis__lot(df_train, df_submission)
    NA_analysis__garages(df_train, df_submission)
    
    printNans(df_train, 'train')
    print(df_train.columns.values)

    printNans(df_submission, 'test')
    print(df_submission.columns.values)

    return df_train, df_submission


def ALZ_garages(df_train, df_submission):
    scatterplotshow(df_train, "YearBuilt", "GarageYrBlt", 'train-ybg')
    scatterplotshow(df_submission, "YearBuilt", "GarageYrBlt", 'test-ybg')

    # typo in GarageYrBlt, 2207 should be 2007
    print(df_submission[df_submission["GarageYrBlt"] >= 2100][["Id", "YearRemodAdd", "YearBuilt", "GarageYrBlt"]])
    df_submission.loc[df_submission["Id"] == 2593, "GarageYrBlt"] = 2007

    print(df_train[["GrLivArea", "LowQualFinSF", "2ndFlrSF", "1stFlrSF", "GarageArea"]].head(5))


# /////////////////////////////////////////////////////////////////////////
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
        #("HasMasonVeneer", lambda row: 1 if row.MasVnrType != "None" else 0), 
        #("HasBasement", lambda row: 1 if row.BsmtQual != "None" else 0), 
        #("Has2ndFloor", lambda row: 1 if row["2ndFlrSF"] > 0.0 else 0), 
        #("HasGarage", lambda row: 1 if row.GarageType != "None" else 0), 
        #("HasWoodDeck", lambda row: 1 if row.WoodDeckSF > 0.0 else 0), 
        #("HasOpenPorch", lambda row: 1 if row.OpenPorchSF > 0.0 else 0), 
        #("HasEnclosedPorch", lambda row: 1 if row.EnclosedPorch > 0.0 else 0), 
        #("Has3SsnPorch", lambda row: 1 if row["3SsnPorch"] > 0.0 else 0), 
        #("HasScreenPorch", lambda row: 1 if row["ScreenPorch"] > 0.0 else 0), 
        #("HasPool", lambda row: 1 if row.PoolQC != "NoPool" else 0), 
        #("HasFence", lambda row: 1 if row.Fence != "NoFence" else 0), 
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
        ("HouseAge", lambda row: row.YrSold - row.YearRemodAdd),
        ("TotalLivArea", lambda row: row.GrLivArea + row.TotalBsmtSF)
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
        ("BsmtQual", {"Ex": 6, "Gd": 5, "TA": 4, "Fa": 3, "Po": 2, "None": 1}),

        # 6 doesn't exist, too little Po.
        ("BsmtCond", {"Ex": 5, "Gd": 5, "TA": 4, "Fa": 3, "Po": 3, "None": 1}),
        ("BsmtExposure", {"Gd": 5, "Av": 4, "Mn": 3, "No": 2, "None": 1}),
        ("BsmtFinType1", {"GLQ": 7, "ALQ": 6, "BLQ": 5, "Rec": 4, "LwQ": 3, "Unf": 2, "None": 1}),
        ("BsmtFinType2", {"GLQ": 7, "ALQ": 6, "BLQ": 5, "Rec": 4, "LwQ": 3, "Unf": 2, "None": 1}),

        # only 1 Po instance, combine with Fa.
        ("HeatingQC", {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 2}),
        ("CentralAir", {"N": 0, "Y": 1}),
        ("KitchenQual", {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}),
        ("FireplaceQu", {"Ex": 6, "Gd": 5, "TA": 4, "Fa": 3, "Po": 2, "None": 1}),
        ("GarageFinish", {"Fin": 4, "RFn": 3, "Unf": 2, "None": 1}),

        # too little Ex and Po.
        ("GarageQual", {"Ex": 5, "Gd": 5, "TA": 4, "Fa": 3, "Po": 3, "None": 1}),
        ("GarageCond", {"Ex": 5, "Gd": 5, "TA": 4, "Fa": 3, "Po": 3, "None": 1}),

        ("PavedDrive", {"Y": 3, "P": 2, "N": 1}),
        ("Condition1", {"Norm":"N", "Feedr": "F", "Artery": "A", "RRAn": "R", "RRAe": "R", "RRNn": "R", "RRNe": "R", "PosN": "P", "PosA": "P"}),
        ("Foundation", {"BrkTil": "BT", "CBlock": "CB", "PConc": "PC", "Slab": "OT", "Stone": "OT", "Wood": "OT"}),
        ("LotConfig", {"Inside": "Inside", "Corner": "Corner", "CulDSac": "CulDSac", "FR2": "FR23", "FR3": "FR23"}),
        ("GarageType", {"Attchd": "Attchd", "Detchd": "Detchd", "BuiltIn": "BuiltIn", "None": "None", "Basment": "Others", "CarPort": "Others", "2Types": "Others" })
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
        "Electrical", "RoofMatl", "LotShape", "LandSlope",
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
    df_forSubs[["Id", "SalePrice"]].to_csv("test.predicted.attempt2.csv", index=False)

    print("Done")



def main():
    df_train = pd.read_csv('train.csv')
    df_submission = pd.read_csv('test.csv')

    # filling NA values:
    #find_na_values(df_train, df_submission)
    df_train, df_submission = PREP_Filling_NA(df_train, df_submission)

    ALZ_garages(df_train, df_submission)

    kfold_train_test(df_train, df_submission)

    

if __name__ == '__main__':
    main()
