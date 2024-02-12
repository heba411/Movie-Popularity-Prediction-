from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X


def outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data.boxplot()  # before removing
    plt.show()
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    data.boxplot()  # after removing
    plt.show()
    return data

def preprocessing(data):
    # Date column
    # data_train = X_train.copy()
    data['release_date'] = pd.to_datetime(data['release_date'])
    data['release_date'] = data['release_date'].dt.year

    # Date column
    data_train = data.copy()
    data_train['release_date'] = pd.to_datetime(data_train['release_date'])
    data['release_date'] = data_train['release_date'].dt.year

    # encoding
    cols = ('genres', 'original_language', 'production_companies', 'production_countries', 'spoken_languages', 'status',
            'homepage',
            'keywords', 'original_title', 'tagline', 'title', 'overview')

    data = Feature_Encoder(data, cols)

    # drop columns with too many nulls
    data_train.drop(columns='homepage', inplace=True)
    # has no effect on data s it has the same value for all rows
    data_train.drop(columns='status', inplace=True)

    data_train["tagline"].fillna(data["tagline"].mean(), inplace=True)
    data_train["runtime"].fillna(data["runtime"].mean(), inplace=True)
    data_train["overview"].fillna(data["overview"].mean(), inplace=True)

    # feature scaling
    data = featureScaling(np.array(data), 0, 10).astype(int)

    return data


