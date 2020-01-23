
import random
import matplotlib.pyplot as plt

def getDsLbl(pX, py, label):
    x = []
    y = []

    for xn,yn in zip(pX, py):
        if yn == label:
            x.append(xn[0])
            y.append(xn[1])

    return x, y

def plotDataset(X, y):
    x1, y1 = getDsLbl(X, y, 1)
    x2, y2 = getDsLbl(X, y, -1)
    plt.scatter(x1, y1, c='red', label="1")
    plt.scatter(x2, y2, c='blue', label="-1")
    plt.show()

def signf(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

def getRandomNumber():
    return random.uniform(-1.0, 1.0)

def getRandomPt():
    return (getRandomNumber(), getRandomNumber())

def generateRandomDatasetCircular(N, noise=0):
    X = []
    y = []

    n_noisy = int(N * noise)
    n_idx = random.sample(range(0, N), n_noisy)

    n = 0
    while (n < N):
        x1, y1 = getRandomPt()

        X.append([x1, y1])
        y.append(signf(x1*x1 + y1*y1 - 0.6))
        n += 1
    
    #plotDataset(m, c, plaDataset)

    for ni in n_idx:
        y[ni] *= -1

    return X, y


def generateRandomDataset(N, m=None, c=None, noise=0):
    """
    N: number of points in dataset
    m: slope of line. If not given, a random slope will be generated
    c: y intercept of line. If not given, also a random number will be generated
    noise: [0.0 - 1.0], the portion of N that sign is flipped, to simulate noise 
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

    n_noisy = int(N * noise)
    n_idx = random.sample(range(0, N), n_noisy)

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

    for ni in n_idx:
        y[ni] *= -1

    return X, y, m, c

