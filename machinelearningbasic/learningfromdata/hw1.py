
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

def mysign(x):
    if x < 0.0:
        return -1
    elif x > 0.0:
        return 1
    else:
        return 0

# Learning problem
class PlaLine:
    def __init__(self, m, c):
        self.m = m
        self.c = c
    
    def y(self, x):
        return self.m * x + self.c

class PlaXpt:
    def __init__(self, x, label=None):
        """
        x: the value vector. Must be a list
        """
        self.x = [1]
        self.x.extend(x) # for PLA, the dataset is prepended with x0 which is always 1
        self.label = label
    
    def __str__(self):
        return "{ values: %s, weights: %s, actualLabel: %s }" % (
            str(self.x), self.label
        )
    
    def h(self, w):
        """
        Obtains raw w \dot x value unabridged
        w: the weight vector
        """
        sum = 0.0
        xl = len(self.x)
        i = 0
        while (i < xl):
            sum += w[i] * self.x[i]
            i += 1
        return sum

    def hsign(self, w):
        """
        Obtains the classification of this point.
        w: the weight vector
        returns: 0, 1, or -1. 0 means Not yet classified
        """
        sum = self.h(w)
        return mysign(sum)

def getRandomNumber():
    return random.uniform(-1.0, 1.0)

def getRandomPt():
    return (getRandomNumber(), getRandomNumber())

def getDsLbl(dataset, label):
    x = []
    y = []

    for ds in dataset:
        if ds.label == label:
            x.append(ds.x[1])
            y.append(ds.x[2])

    return x, y

def plotDataset(m, c, dataset, plaW):
    x1, y1 = getDsLbl(dataset, 1)
    x2, y2 = getDsLbl(dataset, -1)

    plt.scatter(x1, y1, c='red', label="1")
    plt.scatter(x2, y2, c='blue', label="-1")
    plt.plot([-1, 1], [-m + c, m + c], 'g--')
    # plaW = [w0, w1, w2] ==> w0 + x*w1 + y*w2 = 0
    plt.plot([-1, 1], [(plaW[1] - plaW[0]) / plaW[2], (-plaW[1] - plaW[0]) / plaW[2]], 'b:')

    plt.show()

def getProbFG(N, f, g):
    """
    N: number of dataset to be generated
    f: PlaLine, real line
    g: PlaLine, learned line
    """

    numDisagreement = 0

    n = 0
    while n < N:
        x, y = getRandomPt()
        if (mysign(y - f.y(x)) != mysign(y - g.y(x))):
            numDisagreement += 1
        n += 1
    
    return numDisagreement / N

def generateRandomDataset(N):
    """
    N: number of points in dataset
    """

    # generate line, pick 2 random points in the plane, and draw a line
    # any point below that point is considered -1, else +1
    x1,y1 = getRandomPt()
    x2,y2 = getRandomPt()

    # (x1, y1) -line-> (x2, y2)
    # y = mx + c
    # this is **really** bad if x1 == x2. But hey, it should be really rare occurence!
    m = (y1-y2) / (x1-x2)
    c = y1 - m * x1

    plaDataset = []

    # now, the line equation is y = mx+c
    n = 0
    while (n < N):
        x1, y1 = getRandomPt()
        # check if x1,y1 is above / below line
        liney = m * x1 + c
        if (y1 > liney):
            # give +1
            plaDataset.append(PlaXpt([x1, y1], 1))
            n += 1

        elif (y1 < liney):
            # give -1
            plaDataset.append(PlaXpt([x1, y1], -1))
            n += 1
    
    #plotDataset(m, c, plaDataset)

    return plaDataset, m, c

def updatePLAWeights(plaW, yn, xn):
    i = 0
    pl = len(plaW)

    while i < pl:
        plaW[i] += yn * xn[i]
        i += 1


def learnPLA(plaDataset):
    plaW = [0] * 3
    
    numIterations = 0

    while True: 
        misclassified = []
        for ds in plaDataset:
            if (ds.hsign(plaW) != ds.label):
                misclassified.append(ds)

        if len(misclassified) == 0:
            # complete, no more misclassified points
            return plaW, numIterations, misclassified

        ds = misclassified[random.randint(0, len(misclassified)-1)]
        updatePLAWeights(plaW, ds.label, ds.x)

        numIterations += 1

        if numIterations >= 100000: 
            return plaW, numIterations, misclassified

def learn(N, totalIterations):
    sumIterationsUntilConverges = 0
    sumPMisclassified = 0.0
    sumPFdisG = 0.0

    iterNumber = 0
    while (iterNumber < totalIterations):
        plaDataset, m, c = generateRandomDataset(N)
        plaW, numIterations, misclassified = learnPLA(plaDataset)
        #plotDataset(m, c, plaDataset, plaW)
        sumIterationsUntilConverges += numIterations
        sumPMisclassified += len(misclassified) / N

        # generate another N (or as many as required) number of points, check with 
        # f == (y = mx + c) vs g == (y = (-w1/w2)x - w0/w2) 
        sumPFdisG += getProbFG(100, PlaLine(m, c), PlaLine(-plaW[1]/plaW[2], -plaW[0]/plaW[2]))

        iterNumber += 1
    
    return sumIterationsUntilConverges, sumPMisclassified, sumPFdisG

def ans(N, totalIterations):
    si, sp, stp = learn(N, totalIterations)

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
