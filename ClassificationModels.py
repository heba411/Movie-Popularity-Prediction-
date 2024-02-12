import _pickle as pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import time

training_times = []

class Logistic:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        logistic_regression = LogisticRegression(random_state=0)
        start_time_train = time.time()
        logistic_regression.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_time = end_time_train - start_time_train
        training_times.append(training_time)
        filename = 'logistic_model'
        pickle.dump(logistic_regression, open(filename, 'wb'))



class Svc_linear1:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Svc_kernal = SVC(kernel='linear', random_state=0, C=0.000001)
        start_time_train = time.time()
        Svc_kernal.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_time = end_time_train - start_time_train
        training_times.append(training_time)
        filename = 'svc_model'
        pickle.dump(Svc_kernal, open(filename, 'wb'))

class Svc_linear2:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):

        Svc_kernal2 = SVC(kernel='linear', random_state=0, C=1.0)
        start_time_train = time.time()
        Svc_kernal2.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_time = end_time_train - start_time_train
        training_times.append(training_time)
        filename = 'svc_model_hp2'
        pickle.dump(Svc_kernal2, open(filename, 'wb'))

class Svc_linear3:
     def __init__(self, X_train, y_train, X_test, y_test):
                self.X_train = X_train
                self.y_train = y_train
                self.X_test = X_test
                self.y_test = y_test

     def save_model(self):
                Svc_kernal3 = SVC(kernel='linear', random_state=0, C=500)
                start_time_train = time.time()
                Svc_kernal3.fit(self.X_train, self.y_train)
                end_time_train = time.time()
                training_time = end_time_train - start_time_train
                training_times.append(training_time)
                filename = 'svc_model_hp3'
                pickle.dump(Svc_kernal3, open(filename, 'wb'))
class Svc_Polynomial:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        svc_polynomial = SVC(kernel='poly', degree=2)
        start_time_train = time.time()
        svc_polynomial.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_time = end_time_train - start_time_train
        training_times.append(training_time)
        filename = 'polynomial_model'

        pickle.dump(svc_polynomial, open(filename, 'wb'))


class Svc_rbf:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        svc_rbf = SVC(kernel='rbf', gamma=2)
        start_time_train = time.time()
        svc_rbf.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_time = end_time_train - start_time_train
        training_times.append(training_time)
        filename = 'RBF_model'
        pickle.dump(svc_rbf, open(filename, 'wb'))

class Random_forest:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Random_forest = RandomForestClassifier(max_depth=20, criterion='entropy', random_state=0)
        start_time_train = time.time()
        Random_forest.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_time = end_time_train - start_time_train
        training_times.append(training_time)
        filename = 'random_model'
        pickle.dump(Random_forest, open(filename, 'wb'))


class Random_forest_hp2:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Random_forest = RandomForestClassifier(max_depth= 5, criterion='entropy', random_state=0)
        start_time_train = time.time()
        Random_forest.fit(self.X_train,self.y_train)
        end_time_train = time.time()
        training_time = end_time_train - start_time_train
        training_times.append(training_time)
        filename = 'random_model_hp2'
        pickle.dump(Random_forest, open(filename, 'wb'))

class Random_forest_hp3:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Random_forest = RandomForestClassifier(max_depth=2, criterion='entropy', random_state=0)
        start_time_train = time.time()
        Random_forest.fit(self.X_train,self.y_train)
        end_time_train = time.time()
        training_time = end_time_train - start_time_train
        training_times.append(training_time)
        filename = 'random_model_hp3'
        pickle.dump(Random_forest, open(filename, 'wb'))

# class KNN:
#     def __init__(self, X_train, y_train, X_test, y_test):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_test = X_test
#         self.y_test = y_test
#
#     def save_model(self):
#         knn = KNeighborsClassifier(n_neighbors=10)
#         knn.fit(self.X_train, self.y_train)
#
#         filename = 'knn_model'
#         pickle.dump(KNN, open(filename, 'wb'))