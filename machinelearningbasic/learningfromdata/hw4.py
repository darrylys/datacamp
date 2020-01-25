import math
import random
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

def Q1():
    pn = [400000, 420000, 440000, 460000, 480000]
    delta = 0.05
    err = 0.05
    dvc = 10
    diff = math.inf
    bestn = 0

    for n in pn:
        dtmp = 4 * (2 * n)**dvc * math.exp(-1/8*err**2*n)
        if (abs(dtmp - delta) < diff) :
            diff = abs(dtmp - delta)
            bestn = n
    
    print("Q1. bestN = {}".format(bestn))

def Q2(N=10000, dvc=50, delta=0.05):
    """
    Assuming N >> err, the error terms can be ignored here
    """
    ovc = math.sqrt(8 / N * (math.log(4) + dvc * math.log(2*N) - math.log(delta)))
    rademacher = math.sqrt(2 * (math.log(2 * N) + dvc * math.log(N)) / N) + math.sqrt(2 / N * math.log(1 / delta)) + 1/N
    pvb = math.sqrt(1 / N * (math.log(6) + dvc * math.log(2*N) - math.log(delta)))
    devr = math.sqrt(1/(2*N) * (math.log(4) + 2 * dvc * math.log(N) - math.log(delta)))

    print("Q2: Original VC Bound: {}, Rademacher: {}, Parrondo: {}, Devroye: {}".format(
        ovc, rademacher, pvb, devr
    ))

def Q3(N=10, dvc=50, delta=0.05):
    """
    In Q3, the implicit error bound cannot be ignored
    """
    ovc = math.sqrt(8 / N * (math.log(4) + dvc * math.log(2*N) - math.log(delta)))
    rademacher = math.sqrt(2 * (math.log(2 * N) + dvc * math.log(N)) / N) + math.sqrt(2 / N * math.log(1 / delta)) + 1/N
    
    #for below formulas, they are derived from simple quadratic formula.
    #only grab the positive value.
    L = math.log(6) + 2 * N * math.log(2) - math.log(delta)
    pvb = 0.5 * (2 / N + math.sqrt(4 / (N**2) - 4 * (-L / N)))

    Q = math.log(4) + (N**2) * math.log(2) - math.log(delta)
    devr = (2 / N + math.sqrt(4 / (N**2) - 4 * (1 - 2/N)*(-Q/(2*N)))) / (2 * (1 - 2/N))

    print("Q3: Original VC Bound: {}, Rademacher: {}, Parrondo: {}, Devroye: {}".format(
        ovc, rademacher, pvb, devr
    ))


def errsinlin(k, npoints=100):
    esq = 0.0
    i=0
    a = math.sin(math.pi * (k/npoints)) / (k/npoints)
    
    while i <= npoints:
        x = i / npoints
        esq += ((math.sin(math.pi * x) - a * x)**2)
        i += 1
    
    return esq / (npoints+1)

def findmink(npoints):
    i = 1

    a = 0
    minEsq = math.inf
    while i <= npoints:
        etmp = errsinlin(i, npoints)
        if (etmp < minEsq):
            minEsq = etmp
            a = math.sin(math.pi * (i/npoints)) / (i/npoints)
        i += 1
    
    return a

def approxrandom(npoints):
    sum = 0.0

    i = 0
    while i < npoints:
        x = random.uniform(0, 1)
        if x > 0:
            sum += math.sin(math.pi * x) / (x)
            i += 1
    return sum / npoints


def approx4(div):
    div = 10

    sum = 0.0

    i = 1
    while i <= div:
        x = i / div
        sum += math.sin(math.pi * x) / (x)
        i += 1

    return sum / (div)

def generateDataset4(N = 2):
    X = []
    y = []

    _I = 0
    while _I < N:
        xi = random.uniform(-1, 1)
        yi = math.sin(math.pi * xi)
        X.append([xi])
        y.append(yi)
        _I += 1
    
    return X, y

def Q4(totalIterations=10000):
    g = []
    allX = []
    ally = []
    _I = 0
    while _I < totalIterations:
        X, y = generateDataset4()
        allX.append(X)
        ally.append(y)
        linreg = LinearRegression()
        linreg.train(X, y, addYIntercept=False)
        #plt.scatter([X[0][0], X[1][0]], y, c="blue")
        #plt.plot([-1, 1], [-linreg._w[1], linreg._w[1]], "r-")
        #plt.show()
        g.append(linreg._w[1])
        _I += 1
    
    gnp = np.array(g)
    gavg = np.mean(gnp)
    print("avg g = {}".format(gavg))

    return gavg, g, allX, ally

def Q5_6():
    gavg, g, allX, ally = Q4()

    # bias
    sum_bias = 0
    for X, y in zip(allX, ally):
        err_sum = 0
        for xi,yi in zip(X, y):
            y_gavg = gavg*xi[0]
            err_sum += (y_gavg - yi)**2
        sum_bias += err_sum / len(y)
    print("Q5: bias: {}".format(sum_bias / len(ally)))

    # variance
    sum_var = 0
    for gd, X, y in zip(g, allX, ally):
        err_sum = 0
        for xi,yi in zip(X, y):
            y_gavg = gavg*xi[0]
            y_gd = gd*xi[0]
            err_sum += (y_gavg - y_gd)**2
        sum_var += err_sum / len(y)
    print("Q6: var: {}".format(sum_var / len(ally)))

def computeEout(X, y, Xtest, ytest, addYIntercept):
    linreg = LinearRegression()
    linreg.train(X, y, addYIntercept)
    return linreg.regressionTest(Xtest, ytest)

def Q7(totalIterations=1000):
    _I = 0
    eoutlist = []
    while _I < totalIterations:
        X, y = generateDataset4(1000)
        Xtest, ytest = generateDataset4(1000)
        eout = [0]*5

        #0: y = b
        eout[0] = computeEout([[]]*len(y), y, [[]]*len(ytest), ytest, True)
        #1: y = ax
        eout[1] = computeEout(X, y, Xtest, ytest, False)
        #2: y = ax+b
        eout[2] = computeEout(X, y, Xtest, ytest, True)
        #3: y = ax^2
        eout[3] = computeEout([[x[0]**2] for x in X], y, [[x[0]**2] for x in Xtest], ytest, False)
        #4: y = ax^2+b
        eout[4] = computeEout([[x[0]**2] for x in X], y, [[x[0]**2] for x in Xtest], ytest, True)

        eoutlist.append(eout)
        _I += 1
    
    npeout = np.array(eoutlist)
    print("Q7: avg err: {}".format(str(np.mean(npeout, axis=0))))
    

def main():
    pass
    #Q1()
    #Q2(1000000)
    #Q3()
    #Q4()
    #Q5_6()
    Q7()


if __name__ == '__main__':
    main()
