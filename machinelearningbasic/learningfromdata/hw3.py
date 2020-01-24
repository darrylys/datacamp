import math

def N(Pb, err, M):
    return 1.0/(-2.0 * err**2) * math.log(Pb / (2*M))

def main():
    print("Q1: {}".format(N(0.03, 0.05, 1)))
    print("Q2: {}".format(N(0.03, 0.05, 10)))
    print("Q3: {}".format(N(0.03, 0.05, 100)))

if __name__ == '__main__':
    main()
