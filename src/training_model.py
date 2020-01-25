import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn import linear_model


def train(df):
    df = df.dropna()

    predict = 'salary'

    X = np.array(df.drop([predict], 1))
    y = np.array(df[predict])

    best = 0
    for _ in range(50):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

        lm = linear_model.LinearRegression()
        lm.fit(x_train, y_train)
        acc = lm.score(x_test, y_test)

        if acc > best:
            best = acc
            with open('../models/Linear_Regression_Salary.pickle', 'wb') as file:
                pickle.dump(lm, file)

    print(best)
