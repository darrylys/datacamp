
import random

def getRandomNumber():
    return random.uniform(-1.0, 1.0)

def getRandomPt():
    return (getRandomNumber(), getRandomNumber())

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

