from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()

print(digits)

print(digits.keys())
print(digits['DESCR'])

# in Visual Studio Code, this will be told as syntax error (maybe caused by pylinter)
# although it compiles and runs fine
#print(digits.data.shape)
#print(digits.images.shape)

# VSCode has no issues for below.
print(digits['data'].shape)
print(digits['images'].shape)

