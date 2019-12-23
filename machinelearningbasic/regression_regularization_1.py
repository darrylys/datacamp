# src:
# https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
# 
# Regularization main purpose is to reduce complexity of the models generated
# by linear regression when large number variables are introduced
#
# Linear regression is about trying to find the variables in the following equation:
# y' = b0 + b1*x1 + b2*x2 + ... + bn*xn + e ............................................ (1)
# where e is an error variable, a variable that is independent of x and is causing disturbances
# 
# The best 'bi' values are the ones that minimizes the sum difference of y' (y_pred) and y_actual.
# This sum difference is called 'cost function'. Usually, Residual Sum of Squares is used.
#
# Complexity here is defined by the magnitude of the Beta (b) variable in equation (1).
# The magnitude implies the importance the model placed on this particular variable
# The larger the magnitude, the higher the complexity of the model, the less likely the model
# cannot be generalized to real data, which means, more overfitting, less accuracy, etc.
# 
# There are two most popular regularization method, Ridge and Lasso
# Ridge main purpose is to reduce overfitting by minimizing the Beta values (still > 0), 
#     but does not reduce the number of features.
#     With a lot of features, Ridge may be computationally expensive.
# Lasso is generally the method of choice since it can select and remove features, in addition
#     of also doing what Ridge does. Lasso allows for setting the Beta values to zero, effectively
#     removing that feature from model.

#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 8, 6

def lasso_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

def dump_lasso_linear_regression(data):
    #Initialize predictors to all 15 powers of x
    predictors=['x']
    predictors.extend(['x_%d'%i for i in range(2,16)])

    #Define the alpha values to test
    alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

    #Initialize the dataframe to store coefficients
    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
    ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
    coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

    #Define the models to plot
    models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

    fig = plt.figure(figsize=(12,10))

    #Iterate over the 10 alpha values:
    for i in range(10):
        coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i], models_to_plot)

    plt.savefig("{}.png".format('lasso_linear_regression'))
    plt.close(fig)

    return coef_matrix_lasso


def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

def dump_ridge_linear_regression(data):
    #Initialize predictors to be set of 15 powers of x
    predictors=['x']
    predictors.extend(['x_%d'%i for i in range(2,16)])

    #Set the different values of alpha to be tested
    alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

    #Initialize the dataframe for storing coefficients.
    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
    ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
    coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

    models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}

    fig = plt.figure(figsize=(12,10))
    for i in range(10):
        coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)

    plt.savefig("{}.png".format('ridge_linear_regression'))
    plt.close(fig)

    return coef_matrix_ridge


def linear_regression(data, power, models_to_plot):
    #initialize predictors:
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])
    
    #Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])

    #Check if a plot is to be made for the entered power
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for power: %d'%power)

    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret

def dump_std_linear_regression(data):
    #Initialize a dataframe to store the results:
    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
    ind = ['model_pow_%d'%i for i in range(1,16)]
    coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

    #Define the powers for which a plot is required:
    models_to_plot = {
        1:231,
        3:232,
        6:233,
        9:234,
        12:235,
        15:236
    }

    #Iterate through all powers and assimilate results
    fig = plt.figure(figsize=(12,10))
    for i in range(1,16):
        coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
    
    plt.savefig("{}.png".format('vanilla_linear_regression'))
    plt.close(fig)

    return coef_matrix_simple

def generateData():
    """
    Generate random input array with angles from 60deg to 300deg converted to radians
    using 'sin' function
    """

    x = np.array([i*np.pi/180 for i in range(60,300,4)])
    np.random.seed(10)  #Setting seed for reproducability
    y = np.sin(x) + np.random.normal(0,0.15,len(x))
    data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
    return data

def addPolyDataPoints(data):
    """
    src: https://en.wikipedia.org/wiki/Linear_regression
    To model polynomial functions, it is possible to do this by modelling the equation as:
    y = b0 + b1 * x + b2 * x^2
    and set the X as [x, x^2]

    data: Pandas dataframe
    """
    for i in range(2,16):       #power of 1 is already there
        colname = 'x_%d'%i      #new var will be x_power
        data[colname] = data['x']**i
    
    return data

def main():
    data = generateData()
    #plt.plot(data['x'],data['y'],'.')
    #plt.show()
    data = addPolyDataPoints(data)

    pd.options.display.float_format = '{:,.2g}'.format
    #print(dump_std_linear_regression(data))
    #print(dump_ridge_linear_regression(data))
    print(dump_lasso_linear_regression(data))


if __name__ == '__main__':
    main()
