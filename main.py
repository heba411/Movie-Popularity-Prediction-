import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from Pre_processing import *
from models import *
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('movies-regression-dataset.csv')

data = outliers(data)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

#train test split
# In the first step we will split the data in training and remaining dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Preprocessing
X_train = preprocessing(X_train)
X_val = preprocessing(X_val)
X_test = preprocessing(X_test)

# correlation-feature selection
X_train = pd.DataFrame(X_train)
corr = data.corr()
top_feature = corr.index[abs(corr['vote_average']) > 0.05]
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

selector = SelectKBest(f_classif, k=6)
X_train = selector.fit_transform(X_train, y_train)
X_val = selector.transform(X_val)
X_test = selector.transform(X_test)

# ####################### Regression ######################
# features = ['budget','genres','homepage','id','keywords','original_language','overview',
#             'viewercount','production_companies','production_countries','release_date','revenue','runtime',
#             'spoken_languages','status','tagline','title','vote_count']

# Simple linear Regression
LR = linearRegression(X_train, y_train, X_test, y_test, X_val, y_val)
LR.save_model()
lin_model = pickle.load(open('linear_model', 'rb'))
LR_Predicted = lin_model.predict(X_test)
LR_Predicted_val = lin_model.predict(X_val)

print(f"The Mean Squared Error for Linear Regression (Validation) : {metrics.mean_squared_error(y_val, LR_Predicted_val):.2f}")
print(f"The Mean Squared Error for Linear Regression (Test): {metrics.mean_squared_error(y_test, LR_Predicted):.2f}")
print(f"The Mean Absolute Error for Linear Regression (Test) : {metrics.mean_absolute_error(y_test, LR_Predicted):.2f}")
print(f"The R2_Score for Linear Regression (Test) : {metrics.r2_score(y_test, LR_Predicted):.2f}")
print()

plt.scatter(y_test, LR_Predicted)
plt.plot(y_test, y_test, color='red')
plt.xlabel('y-test data')
plt.ylabel('predicted data(linear)')
plt.show()


#Polynomial Regression
poly = polynomial(X_train, y_train, X_test, y_test, X_val, y_val)
X_test_poly, X_val_poly = poly.save_model()

poly_model = pickle.load(open('poly_model', 'rb'))
poly_prediction = poly_model.predict(X_test_poly)
poly_prediction_V = poly_model.predict(X_val_poly)

print(np.shape(X_test_poly))
print(f"The Mean Squared Error for Polynomial Regression (Validation) : {metrics.mean_squared_error(y_val, poly_prediction_V):.2f}")
print(f"The Mean Squared Error for Polynomial Regression (Test) : {metrics.mean_squared_error(y_test, poly_prediction):.2f}")
print(f"The Mean Absolute Error for Polynomial Regression (Test) : {metrics.mean_absolute_error(y_test, poly_prediction):.2f}")
print(f"The R2_Score for Polynomial Regression (Test) : {metrics.r2_score(y_test, poly_prediction):.2f}")

plt.scatter(y_test, poly_prediction)
plt.plot(y_test, y_test, color='red')
plt.xlabel('y-test data')
plt.ylabel('predicted data (polynomial)')
plt.show()

# # Ridge regression
# ridge = Ridge(alpha=0.1)
# ridge.fit(X_new_train, y_train)
#
# # Predictions
# ridge_prediction_V = ridge.predict(X_new_val)
# ridge_prediction = ridge.predict(X_new_test)
#
# print(f"The Mean Squared Error for Ridge regression (Validation) : {metrics.mean_squared_error(y_val, ridge_prediction_V):.2f}")
# print(f"The Mean Squared Error for Ridge regression (Test): {metrics.mean_squared_error(y_test, ridge_prediction):.2f}")
# print(f"The R2_Score for Ridge Regression (Test) : {metrics.r2_score(y_test, ridge_prediction):.2f}")
# print()

def testFile(path):
    if(path):
        test_data = pd.read_csv(path)
        test_data.fillna(test_data.mean(), inplace=True) #filling nan values with mean

        X_testt = test_data.iloc[:, :-1]
        Y_testt = test_data.iloc[:, -1]

        X_testt = preprocessing(X_testt)  # preprocessing
        X_testt = selector.transform(X_testt)  # feature selection

        lin_pred = lin_model.predict(X_testt)
        poly_features = PolynomialFeatures(degree=2)
        X_test_polyy = poly_features.fit_transform(X_testt, Y_testt)
        poly_pred = poly_model.predict(X_test_polyy)

        print('Using The Test File: ')
        print(f"The Mean Square Error for Linear Regression: {metrics.mean_squared_error(Y_testt, lin_pred):.2f}")
        print(f"The R2_Score for Linear Regression: {metrics.r2_score(Y_testt, lin_pred):.2f}")
        print()
        print(f"The Mean Square Error for Polynomial Regression: {metrics.mean_squared_error(Y_testt, poly_pred):.2f}")
        print(f"The R2_Score for Polynomial Regression: {metrics.r2_score(Y_testt, poly_pred):.2f}")

testFile('test_regression.csv')

