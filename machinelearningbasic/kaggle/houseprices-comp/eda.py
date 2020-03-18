
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from sklearn.preprocessing import StandardScaler

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

def fillnas(df):
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

def main():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    df_train = fillnas(df_train)
    #countNulls(df_train, df_test)
    #boxplotPlot(df_train, "SaleCondition", "SalePrice", "dist SaleCondition vs SalePrice")
    #for feat in ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea"]:
    #    scatterplotshow(df_train, feat, "SalePrice", f"{feat} vs SalePrice")
    #distrOfContinous(df_train, "SalePrice", "SalePrice distr")
    #skewcalc(df_train, df_test)
    #fn = lambda row: 1 if row.MasVnrType != "None" else 0
    #df_test.loc[:, "HasMasonVeneer"] = df_test.apply(fn, axis=1)
    #df_train.loc[:, "HasMasonVeneer"] = df_train.apply(fn, axis=1)
    #df_train.loc[:, "SalePrice"] = df_train["SalePrice"].apply(lambda x : math.log(x+1))

    stdc = StandardScaler()
    stdc = stdc.fit(df_train[["SalePrice"]])
    df_train.loc[:, "SalePrice"] = stdc.transform(df_train[["SalePrice"]])
    #print(df_test["OverallQual"].dropna().value_counts())

    smoothedMeanCols = [
        "MSSubClass", "MSZoning", "LotConfig", "Neighborhood", 
        "Condition1", "BldgType", "HouseStyle",
        "RoofStyle", "Exterior1st", "Exterior2nd",
        "MasVnrType", "Foundation", "GarageType", "SaleType", "SaleCondition",
        "Alley", "LandContour", "Fireplaces", "Street", "Fence", 
        "Electrical", "RoofMatl", "LotShape", "LandSlope"
    ]
    #for col in smoothedMeanCols:
    #    boxplotPlot(df_train, col, "SalePrice", title=f"{col}_Vs_SalePrice")
    #scatterplotshow(df_train, "YearBuilt", "SalePrice")

    df_train.loc[:, "YearBuildP20"] = df_train["YearBuilt"].apply(lambda x: x//20)
    print(df_train["YearBuildP20"].value_counts())

    df_train.loc[:, "YearRemodAddP20"] = df_train["YearRemodAdd"].apply(lambda x: x//20)
    print(df_train["YearRemodAddP20"].value_counts())
    
    df_train.loc[:, "GarageYrBltP20"] = df_train["GarageYrBlt"].apply(lambda x: x//20)
    print(df_train["GarageYrBltP20"].value_counts())

    df_test.loc[:, "YearBuildP20"] = df_test["YearBuilt"].apply(lambda x: x//20)
    print(df_test["YearBuildP20"].value_counts())

    df_test.loc[:, "YearRemodAddP20"] = df_test["YearRemodAdd"].apply(lambda x: x//20)
    print(df_test["YearRemodAddP20"].value_counts())

    df_test.loc[:, "GarageYrBltP20"] = df_test["GarageYrBlt"].dropna().apply(lambda x: min(100, max(95, x//20)))
    print(df_test["GarageYrBltP20"].value_counts())


    #boxplotPlot(df_train, "GarageYrBltP20", "SalePrice", "yrts")

    if False:
        oriFeatures = df_train.columns.drop(["Id", "SalePrice"])
        X_train = df_train[oriFeatures].copy()
        y_train = df_train["SalePrice"].copy()
        X_test = df_test[oriFeatures].copy()

        for col in smoothedMeanCols:
            print(f"X_train: {col}")
            print(X_train[col].value_counts())
            print(f"X_test: {col}")
            print(X_test[col].value_counts())

        for col in smoothedMeanCols:
            X_train, X_test = smoothMeanLabelling(X_train, y_train, X_test, col, 'SalePrice', 5)

        Xy_train = X_train.join(y_train)
        al = Xy_train.corr()["SalePrice"]
        sl = [(k,v) for k,v in zip(al.index.values, al.values)]
        sl = sorted(sl, key=lambda x : abs(x[1]), reverse=True)
        for k,v in sl:
            if abs(v) >= 0.05:
                print(f"{k}\t{v}")

    #boxplotPlot(df_train, "MSSubClass", "LotFrontage")
    #modezone = df_train["MSZoning"].mode()
    #print(modezone.values)
    #print(df_train["LotFrontage"].dropna().apply(lambda x: math.sqrt(x)).skew())
    #print(df_test["LotFrontage"].dropna().apply(lambda x: math.sqrt(x)).skew())

    #sns.distplot(df_train["LotArea"].dropna().apply(lambda x: math.log(x+1)), kde=True, label="LotArea")
    #sns.distplot(df_test["LotArea"].dropna().apply(lambda x: math.log(x+1)), kde=True, label="LotArea")
    #plt.legend()
    #plt.show()
    #print(df_train["LotArea"].dropna().apply(lambda x: math.log(x+1)).skew())
    #print(df_test["LotArea"].dropna().apply(lambda x: math.log(x+1)).skew())

    #df_train.loc[:, "LotShapeRI"] = df_train["LotShape"].map({"Reg": "Reg", "IR1": "IR", "IR2": "IR", "IR3": "IR"})
    #boxplotPlot(df_train, "LotShapeRI", "SalePrice")

    # month x year feature does not correlate much with SalePrice.
    # prolly because the timeframe is way too short?
    #df_train.loc[:, "MonthYearSold"] = df_train["YrSold"] * 12 + df_train["MoSold"] - 1
    #scatterplotshow(df_train, "MonthYearSold", "SalePrice")

    #for col in  ["Street", "Alley"]:
    #    print(df_train[col].dropna().value_counts())
    #    print(df_test[col].dropna().value_counts())

    #df_train.loc[:, "HasGarage"] = df_train.apply(lambda row: 1 if row.GarageType != "NoGarage" else 0, axis=1)
    #boxplotPlot(df_train, "HasGarage", "SalePrice")

    #df_train.loc[:, "SalePrice"] = df_train["SalePrice"].apply(lambda x : math.log(x+1))
    #distrOfContinous(df_train, "SalePrice")

    plt.show()
    

def printskew(df_train, df_test):
    skew = df_train.skew(axis=0, skipna=True)
    for k,v in zip(skew.index.values, skew.values):
        if np.abs(v) > 1:
            print(f"Highly SKEWED {k} {v}")
            distrOfContinousNoKDE(df_train, k, f"HIGH SKEW {k} {v}")
        elif np.abs(v) > 0.5 and np.abs(v) <= 1:
            print(f"Moderately skewed {k} {v}")
            distrOfContinousNoKDE(df_train, k, f"Moderately SKEWED {k} {v}")
    

def skewcalc(df_train, df_test):
    # https://help.gooddata.com/doc/en/reporting-and-dashboards/maql-analytical-query-language/maql-expression-reference/aggregation-functions/statistical-functions/predictive-statistical-use-cases/normality-testing-skewness-and-kurtosis
    # |skew| > 1, highly skewed
    # 1 <= |skew| < 0.5, moderately skewed
    # else: roughly normal distribution. 

    printskew(df_train, df_test)

def distrOfContinousNoKDE(df, col, title=""):
    sns.distplot(df[col].dropna(), kde=False, label=col)
    plt.title(title)
    plt.legend()

def distrOfContinous(df, col, title=""):
    sns.distplot(df[col].dropna(), kde=True, label=col)
    plt.title(title)
    plt.legend()

def distplotCategVsContinuous(df, categCol, contiuCol, title=""):
    print(df[categCol].dropna().value_counts())

    uniqueCategs = np.unique(df[categCol].dropna().values)
    for categ in uniqueCategs:
        sns.distplot(df[df[categCol] == categ][contiuCol], kde=True, label=categ)
    
    plt.title(title)
    plt.legend()
    plt.show()

def boxplotPlot(df, x, y, title=""):
    print(df[x].dropna().value_counts())
    fig = plt.figure(figsize=(20, 6))
    plt.tight_layout()
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.legend()
    plt.savefig(f'x{title}.png')
    plt.close(fig)
    #plt.show()

def scatterplotshow(df, x, y, title=""):
    print(f"Corr {x} {y} is {df[x].corr(df[y], method='pearson')}")
    sns.scatterplot(x=x, y=y, data=df)
    plt.title(title)
    plt.legend()
    plt.show()

def LotAreaVsSalePrice(df_train):
    """
    corr with SalePrice = 0.264
    """
    scatterplotshow(df_train, "LotArea", "SalePrice", "LotArea vs SalePrice")

def LotFrontageVsSalePrice(df_train):
    """
    LotFrontage is most likely the width of the house that touches the road.
    if the house is a rectangle, LotFrontage is the width of the house.
    Kinda a regression line, increasing. Pearson corr: 0.352
    """
    scatterplotshow(df_train, "LotFrontage", "SalePrice", "LotFrontage vs SalePrice")

def AlleyVsSalePrice(df_train):
    """
    Gravel path is cheaper, generally.
    """
    boxplotPlot(df_train, "Alley", "SalePrice", "Alley vs SalePrice")

def StreetVsSalePrice(df_train):
    """
    Gravel path is cheaper, generally.
    """
    boxplotPlot(df_train, "Street", "SalePrice", "Street vs SalePrice")

def MSZoningVsSalePrice(df_train):
    """
    C(all) is cheapest
    """
    boxplotPlot(df_train, "MSZoning", "SalePrice", "MSZoning vs SalePrice")

def mssubclassVsSaleprice(df_train):
    """
    different subclass has different mean/value as expected. 30 is the cheapest
    """
    boxplotPlot(df_train, "MSSubClass", "SalePrice", "MSSubClass vs SalePrice")

def countNulls(df_train, df_test):
    # panda assumes NA = NaN. In this dataset, NA does not always mean "Data Not Available"
    # such as Alley. NA in Alley means No Alley Access.

    def showNulls(df):
        df.loc[:,"Alley"] = df["Alley"].fillna("NoAlleyAccess")
        df.loc[:,"BsmtQual"] = df["BsmtQual"].fillna("NoBasement")
        df.loc[:,"BsmtCond"] = df["BsmtCond"].fillna("NoBasement")
        df.loc[:,"BsmtExposure"] = df["BsmtExposure"].fillna("NoBasement")
        df.loc[:,"BsmtFinType1"] = df["BsmtFinType1"].fillna("NoBasement")
        df.loc[:,"BsmtFinType2"] = df["BsmtFinType2"].fillna("NoBasement")
        df.loc[:,"FireplaceQu"] = df["FireplaceQu"].fillna("NoBasement")
        df.loc[:,"GarageType"] = df["GarageType"].fillna("NoGarage")
        df.loc[:,"GarageFinish"] = df["GarageFinish"].fillna("NoGarage")
        df.loc[:,"GarageQual"] = df["GarageQual"].fillna("no garage")
        df.loc[:,"GarageCond"] = df["GarageCond"].fillna("no garage")
        df.loc[:,"PoolQC"] = df["PoolQC"].fillna("no pool")
        df.loc[:,"Fence"] = df["Fence"].fillna("no fence")
        df.loc[:,"MiscFeature"] = df["MiscFeature"].fillna("no misc feature")

        nulls = df.isnull().sum()
        rows = df.shape[0]
        for k,v in zip(nulls.index.values, nulls.values):
            if v > 0:
                print(f"feature={k} missing={v} missing%={v * 100.0 / rows}")

    print("train nulls:")
    showNulls(df_train)

    print("\ntest nulls:")
    showNulls(df_test)

if __name__ == "__main__":
    main()
