from sklearn.datasets import load_iris

iris = load_iris()

# Iris features
print(iris.feature_names)

# Iris names
print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0])

# Lists out all the datasets of an Iris flower
for i in range(len(iris.target)):
    print("Example %d; label %s, features %s" % (i, iris.target[i], iris.data[i]))