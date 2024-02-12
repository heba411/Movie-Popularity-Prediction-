import _pickle as pickle
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


class linearRegression:
    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

    def save_model(self):
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(self.X_train, self.y_train)
        filename = 'linear_model'
        pickle.dump(linear_regression, open(filename, 'wb'))


class polynomial:

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

    def save_model(self):
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(self.X_train)
        X_test_poly = poly_features.transform(self.X_test)
        X_val_poly = poly_features.transform(self.X_val)
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, self.y_train)

        filename = 'poly_model'
        pickle.dump(poly_model, open(filename, 'wb'))
        return X_test_poly, X_val_poly
