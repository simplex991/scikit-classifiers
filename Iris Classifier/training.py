import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree 

iris = load_iris()
test_index = [0,50,100]

# training data
train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis=0)

# testing data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

print(test_data[0], test_target[0])
print("Features: " + str(iris.feature_names))
print("Names: " + str(iris.target_names))
