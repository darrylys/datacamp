#
# sample features: ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# 
# Submissions:
# submission_inclass.csv: 
# > This is the submission from kaggle free course
# > score: 20998.83780
#
# submission_missing_values:
# > Fill missing values with 'mean'
# > score: 16644.83089
#
# Dealing with missing values:
# 1) drop column, pandas: X_train.drop(missingcols, axis=1)
# 2) replace with mean / median
# 3) method 2) + add column "Input_is_missing" with value TRUE / FALSE 
# All three must be tested either way
#
# Dealing with categorical values
# 1) Labelling, replacing categorical values with numeric representation
#    > This will add ordering assumption to those values, which may or may not be appropriate
# 2) One hot encoding, replace categorical values with additional columns:
#    > For example, a column 'Brand' has values: 'Toyota', 'Honda', and 'Daihatsu'
#      The following columns: 'BrandToyota', 'BrandHonda', 'BrandDaihatsu' will be created
#      and the value for column 'BrandToyota' will be '1' if brand == 'Toyota', else '0'.
#      Repeat for all other additional columns
#    > don't use for columns with a lot of categorical values (>= 15)
# As with any machine learning stuff, both must be tested either way, but typically,
# one hot encoding is better. Try this first.
#
#

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = {
    'Quality': [1,  1,  1,  1,  np.nan,  5],
    'Price':   [10, 10, np.nan, 12, 11, 70],
    'Free':    [-1, -2, -3, -4, -5, -6],
    'Label':   [1,  0,  1,  1,  1,  0],
    'City': ['a', 'a', 'b', 'b', 'b', 'c'],
    'Mode': ['x', 'x', 'x', 'x', 'z', 'z']
}
train = pd.DataFrame(data, columns=['Quality', 'Price', 'Free', 'Label'])

X_train = train[['Quality', 'Price', 'Free']]
y_train = train['Label']

print(X_train)

#missingcols = ['Quality', 'Price']
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
inputed_x_train = pd.DataFrame(imr.fit(train[X_train.columns]).transform(train[X_train.columns]))

inputed_x_train.columns = X_train.columns

print(inputed_x_train)
print(inputed_x_train.shape)

ctr = pd.DataFrame(data, columns=['Label', 'City', 'Mode'])

categCols = ['City', 'Mode']
X_ctr = ctr[categCols]
y_ctr = ctr['Label']

le = LabelEncoder()
#X_ctr[categCols] = le.fit_transform(X_ctr[categCols])

# should not be placed below the loop, because X_ctr['City'] has been replaced there.
print(le.fit(X_ctr['City']).transform(['a', 'b', 'b', 'c']))

for col in categCols:
    X_ctr[col] = le.fit(X_ctr[col]).transform(X_ctr[col])

print(X_ctr)

ohe = OneHotEncoder(sparse=False)
complete = pd.DataFrame(data, columns=['Quality', 'Price', 'Free', 'Label', 'City', 'Mode'])
clarified = ohe.fit(complete[categCols]).transform(complete[categCols])
print(complete)
print(clarified)

clarifieddf = pd.DataFrame(clarified, columns=["col%d"%x for x in range(0, len(clarified[0]))])
cc2 = pd.concat([complete, clarifieddf], axis=1)

print(cc2)

ls = [1, 2, 5, 6]
ls2 = [1, 6, 9, 12]

print(list(set(ls).intersection(set(ls2))))


#Solution from kaggle, one contention I have is the 'handle_unknown='ignore'' part
# it was not explained in kaggle course at all, and it begs an explanation
# my solution is basically only grab the intersection of low_cardinality_cols and good_label_cols
# previously.
    # Apply one-hot encoder to each column with categorical data
    #OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    #OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
    #OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

    # One-hot encoding removed index; put it back
    #OH_cols_train.index = X_train.index
    #OH_cols_valid.index = X_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    #num_X_train = X_train.drop(object_cols, axis=1)
    #num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    #OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    #OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

#train = pd.read_csv("train.csv").select_dtypes(exclude=['object'])
#test = pd.read_csv("test.csv").select_dtypes(exclude=['object'])

#nullcols = train.isnull().sum()
#print(nullcols[nullcols > 0])

#print(train[train["LotFrontage"].isnull()])

#missingcols = ['LotFrontage',  'MasVnrArea', 'GarageYrBlt']

#imr = SimpleImputer(missing_values=np.nan, strategy='mean')
#imr = imr.fit(train[missingcols])
#train[missingcols] = imr.transform(train[missingcols])

#print(train[train["LotFrontage"].isnull()])
#print(train[train["MasVnrArea"].isnull()])
#print(train[train["GarageYrBlt"].isnull()])

#train.drop()

# dropping categorical columns.
#drop_X_train = X_train.select_dtypes(['number'])
#drop_X_valid = X_valid.select_dtypes(['number'])

# finding all problematic categorical columns i.e. categorical columns that exists in
# test data but does not exist in training data
    # All categorical columns
    #object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    # Columns that can be safely label encoded
    #good_label_cols = [col for col in object_cols if 
    #                   set(X_train[col]) == set(X_valid[col])]
            
    # Problematic columns that will be dropped from the dataset
    #bad_label_cols = list(set(object_cols)-set(good_label_cols))
            
    #print('Categorical columns that will be label encoded:', good_label_cols)
    #print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)


#train.select_dtypes(['number'])

#train_test_split()


