import pandas as pd
import numpy as np
import pickle
import sklearn
import sklearn.model_selection
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style


class LinearModel:
    def __init__(self, df):
        self.df = df
        predict = 'salary'

        X = np.array(df.drop([predict], 1))
        y = np.array(df[predict])

        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                                        test_size=0.3)

    def train(self):
        self.model = sm.OLS(self.y_train, self.x_train).fit()

        with open('../models/Linear_Regression_Salary.pickle', 'wb') as file:
            pickle.dump(self.model, file)

        return self.model

    def test(self):
        pickle_in = open('../models/Linear_Regression_Salary.pickle', 'rb')
        lm = pickle.load(pickle_in)

        predictions = self.predict(self.x_test)

        prediction_table = {
            'Predicted Salary': predictions,
            'Actual Value': self.y_test
        }

        return pd.DataFrame(prediction_table)

    def predict(self, value):
        pickle_in = open('../models/Linear_Regression_Salary.pickle', 'rb')
        lm = pickle.load(pickle_in)

        predictions = lm.predict(value)

        return predictions

    def scatterplot(self, comparer):
        style.use('ggplot')
        sns.lmplot(comparer, 'salary', data=self.df)
        plt.show()

    def histplot(self, comparer):
        style.use('ggplot')
        plt.hist(self.df[comparer])
        plt.xlabel(comparer)
        plt.show()

    def summary(self):
        return self.model.summary()
