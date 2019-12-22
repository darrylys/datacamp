# src: https://medium.com/datadriveninvestor/regression-in-machine-learning-296caae933ec
# Regression models are used to predict a continuous value. 
# Sample: Predicting prices of a house given the features of house like size, 
# price etc is one of the common examples of Regression. It is a supervised technique. 

# types:
# 1. Simple Linear Regression
# 2. Polynomial Regression
# 3. Support Vector Regression
# 4. Decision Tree Regression
# 5. Random Forest Regression

# src2: https://www.geeksforgeeks.org/regression-classification-supervised-machine-learning/

# lets check life_expectancy_years.csv data from Gapminder data

import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("hdi_human_development_index.csv")

df.set_index('country', inplace=True)

country = 'Brazil'
bdf = df.loc[country]

# src: https://www.geeksforgeeks.org/how-to-get-column-names-in-pandas-dataframe/
Xo = [int(x) for x in df.columns]
yo = bdf.values

plt.scatter(Xo, yo,  color='black') 
plt.title('HDI of {}'.format(country)) 
plt.xlabel('Year') 
plt.ylabel('HDI') 

# X should be in format [ [<sample 1 features>], [<sample 2 features>], ... ]
# instead of, just a 1D array. Of course, the below should be implemented using
# reshape, but since Xo is just a list, it does not have that. Thus, must do it myself!
X = [[x] for x in Xo]
y = yo

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

lgr = linear_model.LinearRegression()
lgr.fit(X_train, y_train)

plt.plot(X_test, lgr.predict(X_test), color='red',linewidth=2) 
plt.show()
