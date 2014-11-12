print __doc__

from sklearn import datasets, neighbors, linear_model

digits      = datasets.load_digits()
x_digits    = digits.data
y_digits    = digits.target

n_samples = len(x_digits)

x_train = x_digits[: .9 * n_samples]
y_train = y_digits[: .9 * n_samples]
x_test  = x_digits[.9 * n_samples :]
y_test  = y_digits[.9 * n_samples :]

knn         = neighbors.KNeighborsClassifier()
logistic    = linear_model.LogisticRegression()

print('KNN score: %f' % knn.fit(x_train, y_train).score(x_test, y_test))
print('LogisticRegression score: %f' % logistic.fit(x_train, y_train).score(x_test, y_test))