import math
import random

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

def Q4():
    print("100. {}".format(findmink(100)))
    print("500. {}".format(findmink(500)))
    print("1000. {}".format(findmink(1000)))
    print("2000. {}".format(findmink(2000)))

def main():
    pass
    #Q1()
    #Q2(1000000)
    #Q3()

if __name__ == '__main__':
    main()
