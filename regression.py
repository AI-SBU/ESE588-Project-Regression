import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor  # dealing with multicollinearity

sns.set()
"""

Loading the data into Pandas Dataframe
Using Pandas function get the categorical features
read for use
Categorical features used: RAD, CHAS
"""
df = pd.read_csv("BostonHousingData.csv")  # loading the csv file
dummies = pd.get_dummies(df["RAD"], drop_first=True)  # removing the first level reduces multicollinearity
df = df.join(dummies)  # merging the dataframes
df = df.drop(columns=["RAD"])  # dropping the original categorical variable columns
dummies = pd.get_dummies(df["CHAS"], drop_first=True)
df = df.join(dummies)
df = df.drop(columns=["CHAS"])

"""

The function below takes in two parameter, single column dataframes, and plots
their distribution. It is used to check the distribution of the actual prices
vs the predicted prices
"""


def distribution_plot(actual, predicted):
    y_vals = pd.concat([actual, predicted])  # merges the data into a single dataframe
    y_vals.plot.kde()   # plots kernel distribution plot
    plt.title("MEDV Distribution : Actual vs Predicted")
    plt.xlabel("MEDV")
    plt.savefig("distribution.png")
    plt.show()



"""

The function below is used to find multicollinearity using the variance inflation factor (VIF). A high values of
VIF for two or more features suggests that they are highly correlated
The parameter represents the dataframe that containing all of the features
@:parameter boston_data_two
"""


def find_multicollinearity(boston_data_one):
    vif = pd.DataFrame()
    vif["FEATURES"] = boston_data_one.columns   # assigning feature
    # calculating the VIF using the variance_inflation_factor function from statsmodels package
    vif["VIF"] = [variance_inflation_factor(boston_data_one.values, i) for i in range(len(boston_data_one.columns))]
    # vif.to_excel("mlr_VIF.xlsx")
    print(vif)


"""
The function below constructs an MLR model using features indicated in
x_names list
The parameter represents the dataframe that containing all of the features
@:parameter boston_data_two
"""


def multiple_lr(boston_data_two):
    x_names = ["CRIM", "RM", "PTRATIO", "LSTAT", "NOX", "TAX", 1, 2, 3, 4, 5, 6, 7, 8, 24]  # independent vars
    # list
    y = boston_data_two[["MEDV"]]
    x = boston_data_two[x_names]
    x = sm.add_constant(x)  # adding intercept
    mlr = sm.OLS(y, x).fit()  # fitting the model
    yhat = mlr.predict(x)  # predicting all of the y values
    yhat = pd.DataFrame(yhat)
    yhat.columns = ["Predicted MEDV"]
    with open("summary.txt", "w") as file:
        file.write(mlr.summary(alpha=.05).as_text())  # printing the summary to a file, ALPHA is set to .05
    distribution_plot(y, yhat)


multiple_lr(df)
# find_multicollinearity(df)
