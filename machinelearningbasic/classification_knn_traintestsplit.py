from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

colheaders = ['A', 'B', 'C', 'D', 'PETALS']
data = pd.read_csv("iris.data", names=colheaders)
X = data[colheaders[0:-1]]
y = data[colheaders[-1]]

knn = KNeighborsClassifier(n_neighbors = 6)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

