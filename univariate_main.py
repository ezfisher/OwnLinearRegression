from linear_regression import UnivariateLinearRegression
from sklearn.linear_model import LinearRegression
from load_data import x_train, x_test, y_train, y_test

scores = []

my_clf = UnivariateLinearRegression()
my_clf.fit(x_train, y_train)
pred = my_clf.predict(x_test)
scores.append(pred, y_test)

sklearn_clf = LinearRegression()
sklearn_clf.fit(x_train, y_train)
sklearn_pred = sklearn_clf.predict(x_test)
scores.append(x_test, y_test)

print(scores)