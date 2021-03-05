import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

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
    with open("summary.txt", "w") as file:
        file.write(mlr.summary(alpha=.05).as_text())
    # distribution_plot(y, yhat)


multiple_lr()
