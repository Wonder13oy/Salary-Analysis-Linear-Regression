import pandas as pd
import numpy as np
import pickle
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style

lm = None

def test(df):

    df = df.dropna()

    predict = 'salary'

    X = np.array(df.drop([predict], 1))
    y = np.array(df[predict])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    pickle_in = open('../models/Linear_Regression_Salary.pickle', 'rb')
    lm = pickle.load(pickle_in)

    print(f'Coefficient: {lm.coef_}')
    print(f'Intercept: {lm.intercept_}')

    predictions = lm.predict(x_test)

    prediction_table = {
        'Predicted Salary': predictions,
        'Actual Value': y_test
    }

    return pd.DataFrame(prediction_table)


def scatterplot(df, comparer):

    style.use('ggplot')
    sns.lmplot(comparer, 'salary', data=df)
    plt.show()


def histplot(df, comparer):

    style.use('ggplot')
    plt.hist(df['salary'])
    plt.xlabel(comparer)
    plt.show()
