import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from scipy import stats
import math

sns.set()

df = pd.read_csv("BostonHousingData.csv")  # loading the csv file


# print(df.head())
# df = df._get_numeric_data() # working with only numeric data

def multiple_lr():
    # print("\nMLR Model \n")
    mlr = LinearRegression()
    y = df[["MEDV"]]
    x = df[["CRIM", "ZN", "NOX", "RM", "AGE", "RAD", "PTRATIO", "LSTAT"]]
    mlr.fit(x, y)
    yhat = mlr.predict(x)
    x_coef = mlr.coef_
    x_list = []
    for l in x_coef:
        for name in l:
            x_list.append(name) # 1d list
    x_names = ["CRIM", "ZN", "NOX", "RM", "AGE", "RAD", "PTRATIO", "LSTAT"]
    dict_one = {x_names[i]: x_list[i] for i in
                range(len(x_names))}  # making dict with names as keys and coef as values
    dict_two = {"Y-Intercept : ": mlr.intercept_, "R-Square : ": mlr.score(x, y),
                "RMSE : ": math.sqrt(mse(df["MEDV"], yhat))}

    dict_merged = {**dict_one, **dict_two}
    mlr_data = pd.DataFrame(dict_merged)
    print(mlr_data)
    # mlr_data.to_excel("mlr_data.xlsx")


multiple_lr()
