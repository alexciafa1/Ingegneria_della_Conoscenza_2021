import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


def getData(path):
    data = pd.read_csv(path)
    datas = data.drop('Age', 1)
    X_data = np.array(datas.drop('Heart Disease', 1))
    y_data = np.array(data['Heart Disease'])
    n_class = datas.drop_duplicates(subset='Heart Disease')['Heart Disease']
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle='true',stratify=y_data)
    return X_train, X_test, Y_train, Y_test, n_class
