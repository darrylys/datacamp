
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

OUTPREFIX = "eda3"

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

def eda3(df_train, df_test):
    
    # find numerical variables highly correlated to SalePrice
    corr = df_train.corr()

    print("Features with corr SalePrice >= 0.5")
    corrToSalePrice = corr["SalePrice"]

    corrToSalePriceHigh = []
    for k,v in zip(corrToSalePrice.index.values, corrToSalePrice.values):
        if abs(v) >= 0.5:
            print(f"{k}\t{v}")
            corrToSalePriceHigh.append(k)
    
    # it can be seen the presence of highly correlated features
    # corr score >= 0.8
    heatmap(df_train[corrToSalePriceHigh].corr(), "HeatMapGTE0.5")

    # the best score, 0.79 correlation
    boxplotPlot(df_train, "OverallQual", "SalePrice", "OverallQual_vs_SalePrice")

    # second best, 0.70. There are 2 possible outliers, Id: 524 and 1299.
    # and also, these two has highest OverallQual.
    scatterplotshow(df_train, "GrLivArea", "SalePrice", "GrLivArea_vs_SalePrice")
    print(df_train[df_train["GrLivArea"] > 4500][["Id", "SalePrice", "OverallQual"]])

    goodNumerics = ["OverallQual","YearBuilt","YearRemodAdd","TotalBsmtSF",
            "1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","GarageCars","GarageArea"]
    # possibly removed due to high correlation:
    # 1stFlrSF or TotalBsmtSF (0.82)
    # GrLivArea or TotRmsAbvGrd (0.83)
    # GarageCars or GarageArea (0.89)

    categoricals = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",
        "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
        "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", 
        "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", 
        "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", 
        "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", 
        "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", 
        "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]

    # check pool, no missing data in PoolQC
    print(df_train[(df_train["PoolQC"] == "NoPool") & (df_train["PoolArea"] > 0)])

    # there are missing data in Pool, PoolQC is NoPool but PoolArea is greater than 0.
    print(df_test[(df_test["PoolQC"] == "NoPool") & (df_test["PoolArea"] > 0)])

    df_train.loc[:, "PoolQC"] = df_train["PoolQC"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "NoPool": 1})
    df_test.loc[:, "PoolQC"] = df_test["PoolQC"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "NoPool": 1})

    # the corr here seems pretty high (both ~0.7)
    print(df_train[["PoolQC", "PoolArea"]].corr())
    print(df_test[["PoolQC", "PoolArea"]].corr())

    # Fence does not seem to be ordinal
    # see the vFence_Vs_SalePriceStd.png image, it can be seen that NoFence is somehow the highest SalePrice!

    # Fireplaces and FireplaceQu
    # check the |Fireplaces| == 0 and FireplaceQu == NoFireplace
    print(df_train["Fireplaces"].value_counts())
    print(df_train["FireplaceQu"].value_counts())
    print(df_test["Fireplaces"].value_counts())
    print(df_test["FireplaceQu"].value_counts())
    # for both train and test, |Fireplaces| == 0 and |FireplaceQu == NoFireplace| matches. Therefore,
    # NA in FireplaceQu can be safely replaced with NoFireplace

    # Lot related properties
    # LotFrontage, LotArea, LotShape, LotConfig
    # fill LotFrontage with median from neighborhood
    # LotShape, Regular is the cheapest. Seems to be ordinal.
    boxplotPlot(df_train, "LotShape", "SalePrice", "LotShape_vs_SalePrice")
    # LotArea is numeric and is already removed because of the correlation (~0.2) treshold.
    # LotConfig
    print(df_train["LotConfig"].value_counts())

    # Garage

    # general plan:
    # Fix the inputs by comparing related features such as Garage* features.

    # remove utilities because there's only one non AllPub in train, and
    # only NA remaining in test. If in test, inputting AllPub, means this feature is worthless.
    # remove Utilities.

    # fill NAs in a more careful way here. They must be dealt in groups if any. Else
    # just fill with modes / medians which is the standard. BUT, must make sure that
    # when filling with modes / medians, only use the train set.

    # checking ordinality by using violinplot to SalePrice.

    # dwelling variables: HouseStyle and BuildingType
    # conditions variable: whether a house is within vicinity of "conditions": Neighborhood, Condition1 and 2

    # Street and pavement driveway
    pass

if __name__ == '__main__':
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    df_train = replaceNAInData(df_train)
    df_test = replaceNAInData(df_test)

    eda3(df_train, df_test)
