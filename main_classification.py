import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from Pre_processing import *
from sklearn.preprocessing import LabelEncoder
from ClassificationModels import *


data = pd.read_csv('movies-classification-dataset.csv')
data = outliers(data)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Preprocessing
X_train = preprocessing(X_train)
X_test = preprocessing(X_test)

X_train[X_train < 0] = 0
X_test[X_test < 0] = 0

# #Rate encoding
# le = LabelEncoder()
# le.fit(data['Rate'])
# data['Rate'] = le.transform(data['Rate'])

#Feature_Selection
selector = SelectKBest(score_func=chi2, k=6)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)
# data.corr()['Rate'].sort_values(ascending=False).plot(kind='bar')
# plt.show()

testing_times = []

#Logistic Regression
LR = Logistic(X_train, y_train, X_test, y_test)
LR.save_model()
logistic_model = pickle.load(open('logistic_model', 'rb'))
start_time_test = time.time()
LR_Predicted = logistic_model.predict(X_test)
end_time_test = time.time()
testing_time = end_time_test - start_time_test
testing_times.append(testing_time)
print(f"The Accuracy for Logistic Regression: {metrics.accuracy_score(y_test, LR_Predicted):.2f}")

#SVC linear kernel
svc_linear = Svc_linear1(X_train, y_train, X_test, y_test)
svc_linear.save_model()
svc_linear_model = pickle.load(open('svc_model','rb'))
start_time_test = time.time()
linear_predicted = svc_linear_model.predict(X_test)
end_time_test = time.time()
testing_time = end_time_test - start_time_test
testing_times.append(testing_time)

print(f"The Accuracy for SVC linear kernel 1: {metrics.accuracy_score(y_test, linear_predicted):.2f}")

#SVC linear kernel
svc_linear2 = Svc_linear2(X_train, y_train, X_test, y_test)
svc_linear2.save_model()
svc_linear_model2 = pickle.load(open('svc_model_hp2','rb'))
start_time_test = time.time()
linear_predicted2 = svc_linear_model2.predict(X_test)
end_time_test = time.time()
testing_time = end_time_test - start_time_test
testing_times.append(testing_time)

print(f"The Accuracy for SVC linear kernel 2: {metrics.accuracy_score(y_test, linear_predicted2):.2f}")

#SVC linear kernel
svc_linear3 = Svc_linear3(X_train, y_train, X_test, y_test)
svc_linear3.save_model()
svc_linear_model3 = pickle.load(open('svc_model_hp3','rb'))
start_time_test = time.time()
linear_predicted3 = svc_linear_model3.predict(X_test)
end_time_test = time.time()
testing_time = end_time_test - start_time_test
testing_times.append(testing_time)

print(f"The Accuracy for SVC linear kernel 3: {metrics.accuracy_score(y_test, linear_predicted3):.2f}")



# #SVC rbf kernel
svc_rbf = Svc_rbf(X_train, y_train, X_test, y_test)
svc_rbf.save_model()
svc_rbf_model = pickle.load(open('RBF_model','rb'))
start_time_test = time.time()
rbf_predicted = svc_rbf_model.predict(X_test)
end_time_test = time.time()
testing_time = end_time_test - start_time_test
testing_times.append(testing_time)

print(f"The Accuracy for SVC RBF kernel: {metrics.accuracy_score(y_test, rbf_predicted):.2f}")

#SVC poly kernel
svc_poly = Svc_Polynomial(X_train, y_train, X_test, y_test)
svc_poly.save_model()
svc_poly_model = pickle.load(open('polynomial_model','rb'))
start_time_test = time.time()
poly_predicted = svc_poly_model.predict(X_test)
end_time_test = time.time()
testing_time = end_time_test - start_time_test
testing_times.append(testing_time)

print(f"The Accuracy for SVC poly kernel: {metrics.accuracy_score(y_test, poly_predicted):.2f}")

#Random Forest1
RF = Random_forest(X_train, y_train, X_test, y_test)
RF.save_model()
RF_model = pickle.load(open('random_model','rb'))
start_time_test = time.time()
RF_predicted = RF_model.predict(X_test)
end_time_test = time.time()
testing_time = end_time_test - start_time_test
testing_times.append(testing_time)

print(f"The Accuracy for Random Forest 1: {metrics.accuracy_score(y_test, RF_predicted):.2f}")

#Random Forest2
RF2 = Random_forest_hp2(X_train, y_train, X_test, y_test)
RF2.save_model()
RF_model2 = pickle.load(open('random_model_hp2','rb'))
start_time_test = time.time()
RF_predicted2 = RF_model2.predict(X_test)
end_time_test = time.time()
testing_time = end_time_test - start_time_test
testing_times.append(testing_time)

print(f"The Accuracy for Random Forest 2: {metrics.accuracy_score(y_test, RF_predicted2):.2f}")

#Random Forest3
RF3 = Random_forest_hp3(X_train, y_train, X_test, y_test)
RF3.save_model()
RF_model3 = pickle.load(open('random_model_hp3','rb'))
start_time_test = time.time()
RF_predicted3 = RF_model3.predict(X_test)
end_time_test = time.time()
testing_time = end_time_test - start_time_test
testing_times.append(testing_time)

print(f"The Accuracy for Ranfdom Forest 3: {metrics.accuracy_score(y_test, RF_predicted3):.2f}")


models = ['logistic', 'svmLin1','svmLin2','svmLin3', 'svm_rbf', 'svmPoly', 'RF1','RF2', 'RF3']
plt.bar(models, training_times, color='lightblue', label='Training Time')
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title('Training Times for Each Model')
plt.legend()
plt.show()

models_test = ['logistic', 'svmLin1','svmLin2','svmLin3', 'svm_rbf', 'svmPoly', 'RF1','RF2', 'RF3']
plt.bar(models_test, testing_times, color='limegreen', label='Testing Time')
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title(' Testing Times for Each Model')
plt.legend()
plt.show()

accuracy_scores = [metrics.accuracy_score(y_test, LR_Predicted), metrics.accuracy_score(y_test, linear_predicted),
 metrics.accuracy_score(y_test, linear_predicted2),metrics.accuracy_score(y_test, linear_predicted3),
 metrics.accuracy_score(y_test, rbf_predicted), metrics.accuracy_score(y_test, poly_predicted),
 metrics.accuracy_score(y_test, RF_predicted), metrics.accuracy_score(y_test, RF_predicted2),
 metrics.accuracy_score(y_test, RF_predicted3)]

models_acc = ['logistic', 'svmLin 1','svmLin 2','svmLin 3', 'svm_rbf', 'svmPoly', 'RF1','RF2', 'RF3']
plt.bar(models_acc, accuracy_scores, color='pink')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy for Each Model')

# Display the accuracy values on top of each bar
for i, v in enumerate(accuracy_scores):
    plt.text(i, v, str(round(v, 2)), ha='center', va='bottom')

plt.show()

def testFile(path):
    if(path):
        test_data = pd.read_csv(path)
        test_data.fillna(test_data.mean(), inplace=True) #filling nan values with mean
        # # Rate encoding
        # lE = LabelEncoder()
        # lE.fit(test_data['Rate'])
        # test_data['Rate'] = lE.transform(test_data['Rate'])

        X_testt = test_data.iloc[:, :-1]
        Y_testt = test_data.iloc[:, -1]

        X_testt = preprocessing(X_testt)  # preprocessing
        # X_testt[X_testt < 0] = 0
        X_testt = selector.transform(X_testt)  # feature selection


        log_pred = logistic_model.predict(X_testt)
        svcLinear_pred = svc_linear_model.predict(X_testt)
        svcLinear_pred2 = svc_linear_model2.predict(X_testt)
        svcLinear_pred3 = svc_linear_model3.predict(X_testt)
        svcRBF_pred = svc_rbf_model.predict(X_testt)
        svcPoly_pred = svc_poly_model.predict(X_testt)
        RandomF_pred = RF_model.predict(X_testt)
        Random2 = RF_model2.predict(X_testt)
        Random3 = RF_model3.predict(X_testt)


        print('Using The Test File:')
        print(f"The Accuracy for Logistic Regression: {metrics.accuracy_score(Y_testt, log_pred):.2f}")
        print(f"The Accuracy for SVC Linear Kernel 1: {metrics.accuracy_score(Y_testt, svcLinear_pred):.2f}")
        print(f"The Accuracy for SVC Linear Kernel 2: {metrics.accuracy_score(Y_testt, svcLinear_pred2):.2f}")
        print(f"The Accuracy for SVC Linear Kernel 3: {metrics.accuracy_score(Y_testt, svcLinear_pred3):.2f}")
        print(f"The Accuracy for SVC RBF Kernel: {metrics.accuracy_score(Y_testt, svcRBF_pred):.2f}")
        print(f"The Accuracy for SVC Poly Kernel: {metrics.accuracy_score(Y_testt, svcPoly_pred):.2f}")
        print(f"The Accuracy for Random Forest 1: {metrics.accuracy_score(Y_testt, RandomF_pred):.2f}")
        print(f"The Accuracy for Random Forest 2: {metrics.accuracy_score(Y_testt, Random2):.2f}")
        print(f"The Accuracy for Random Forest 3: {metrics.accuracy_score(Y_testt, Random3):.2f}")


testFile(' D:/myProjects/Machine_MS2_2/Machine_MS2_2/Machine_MS2/day1/M2/movies-tas-test')












