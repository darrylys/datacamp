
import random
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
import sys

def getRandomNumber():
    return random.uniform(-1.0, 1.0)

def getRandomPt():
    return (getRandomNumber(), getRandomNumber())

def getDsLbl(X, y, label):
    x = []
    y = []

    for xn,yn in zip(X, y):
        if yn == label:
            x.append(xn)
            y.append(yn)

    return x, y

def plotDataset(m, c, X, y, plaW):
    x1, y1 = getDsLbl(X, y, 1)
    x2, y2 = getDsLbl(X, y, -1)

    plt.scatter(x1, y1, c='red', label="1")
    plt.scatter(x2, y2, c='blue', label="-1")
    plt.plot([-1, 1], [-m + c, m + c], 'g--')
    # plaW = [w0, w1, w2] ==> w0 + x*w1 + y*w2 = 0
    plt.plot([-1, 1], [(plaW[1] - plaW[0]) / plaW[2], (-plaW[1] - plaW[0]) / plaW[2]], 'b:')

    plt.show()

def generateRandomDataset(N, m=None, c=None):
    """
    N: number of points in dataset
    m: slope of line. If not given, a random slope will be generated
    c: y intercept of line. If not given, also a random number will be generated
    If either of m or c is not defined, both will be generated. Remember this!
    """

    # generate both if either not defined
    if not m or not c:
        # generate line, pick 2 random points in the plane, and draw a line
        # any point below that point is considered -1, else +1
        x1,y1 = getRandomPt()
        x2,y2 = getRandomPt()

        # (x1, y1) -line-> (x2, y2)
        # y = mx + c
        # this is **really** bad if x1 == x2. But hey, it should be really rare occurence!
        m = (y1-y2) / (x1-x2)
        c = y1 - m * x1

    X = []
    y = []

    # now, the line equation is y = mx+c
    n = 0
    while (n < N):
        x1, y1 = getRandomPt()
        # check if x1,y1 is above / below line
        liney = m * x1 + c
        if (y1 > liney):
            # give +1
            X.append([x1, y1])
            y.append(1)
            n += 1

        elif (y1 < liney):
            # give -1
            X.append([x1, y1])
            y.append(-1)
            n += 1
    
    #plotDataset(m, c, plaDataset)

    return X, y, m, c

def learn(N, totalIterations):
    sumIterationsUntilConverges = 0
    sumPFdisG = 0.0

    iterNumber = 0
    while (iterNumber < totalIterations):
        X, y, m, c = generateRandomDataset(N)
        pla = Perceptron()
        pla.train(X, y)
        
        sumIterationsUntilConverges += pla._numIterations

        # generate another N (or as many as required) number of points, check with 
        # f == (y = mx + c) vs g == (y = (-w1/w2)x - w0/w2)
        X_test, y_test, m, c = generateRandomDataset(100, m, c)
        sumPFdisG += pla.test(X_test, y_test) 

        iterNumber += 1
    
    return sumIterationsUntilConverges, sumPFdisG

def ans(N, totalIterations):
    si, stp = learn(N, totalIterations)

    print("N: %d, totalIterations: %d" % (N, totalIterations))
    print("avg Iterations: %f" % (si / totalIterations))
    print("avg P(f!=g): %f" % (stp / totalIterations))

def main():
    # generate a random line in [-1,1]x[-1,1]
    # pick 2 random dots
    
    N = 100
    totalIterations = 1000
    if (len(sys.argv) >= 3):
        N = int(sys.argv[1])
        totalIterations = int(sys.argv[2])
        ans(N, totalIterations)

    else:
        ans(10, 1000)
        ans(100, 1000)


if __name__ == '__main__':
    main()
