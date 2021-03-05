import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy import stats
import math

sns.set()


def distribution_plot(actual, predicted):
    y_vals = pd.concat([actual, predicted])
    y_vals.plot.kde()
    plt.title("MEDV Distribution : Actual vs Predicted")
    plt.xlabel("MEDV")
    plt.show()


def auction_example_book():
    data = pd.read_csv("auction.txt", sep="\t")
    # print(data)
    x_names = ["AGE", "NUMBIDS"]  # independent vars list
    y = data[["PRICE"]]
    x = data[x_names]
    mlr = sm.OLS(y, x)
    mlr = mlr.fit()
    yhat = mlr.predict(x)
    data["yhat"] = yhat
    data.drop(columns=["AGE-BID"], inplace=True)
    fval, pval = stats.f_oneway(data["AGE"], data["NUMBIDS"])
    print("F-test ", fval, "\nP-val : ", pval)
    # print(data)


def multiple_lr():
    df = pd.read_csv("BostonHousingData.csv")  # loading the csv file
    x_names = ["CRIM", "ZN", "INDUS", "NOX", "RM", "TAX", "PTRATIO", "LSTAT"]  # independent vars list
    y = df[["MEDV"]]
    x = df[x_names]
    mlr = sm.OLS(y, x).fit()
    yhat = mlr.predict(x)
    yhat = pd.DataFrame(yhat)
    yhat.columns = ["Predicted MEDV"]
    # print(yhat)
    # print(mlr.summary())
    distribution_plot(y, yhat)


multiple_lr()
