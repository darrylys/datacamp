from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def sumdiff(A, B):
    L = len(A)
    I = 0
    sum = 0
    while (I < L):
        if A[I] == B[I]:
            sum += 1
        I += 1
    return sum

colheaders = ['A', 'B', 'C', 'D', 'PETALS']
data = pd.read_csv("iris.data", names=colheaders)

data.info()
#print(data[colheaders[0:-1]])

X = data[colheaders[0:-1]]
y = data[colheaders[-1]]

knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X, y)

predicted = knn.predict(X)

correct = sumdiff(predicted, y)
L = len(predicted)

print("train accuracy (0,1): {}".format(correct / L))


