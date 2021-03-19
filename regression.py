import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor  # dealing with multicollinearity
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score)
from sklearn.metrics import classification_report

sns.set()

"""

The function below takes in two parameter, single column dataframes, and plots
their distribution. It is used to check the distribution of the actual prices
vs the predicted prices
"""


def distribution_plot(actual, predicted, name):
    y_vals = pd.concat([actual, predicted])  # merges the data into a single dataframe
    y_vals.plot.kde()  # plots kernel distribution plot
    plt.title("Actual vs Predicted")
    plt.xlabel(name)
    plt.savefig("distribution.png")
    plt.show()


def count_plot(column_name, dataframe):
    sns.countplot(x=column_name, data=dataframe)
    plt.title("Outcome: Positive vs Negative")
    plt.show()


"""

The function below is used to find multicollinearity using the variance inflation factor (VIF). A high values of
VIF for two or more features suggests that they are highly correlated
The parameter represents the dataframe that containing all of the features
@:parameter boston_data_two
"""


def find_multicollinearity(boston_data_one):
    vif = pd.DataFrame()
    vif["FEATURES"] = boston_data_one.columns  # assigning feature
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


def multiple_lr(x, y, yhat_name, file_path, response_var):
    x = sm.add_constant(x)  # adding intercept
    mlr = sm.OLS(y, x).fit()  # fitting the model
    yhat = mlr.predict(x)  # predicting all of the y values
    yhat = pd.DataFrame(yhat)
    with open(file_path, "w") as file:
        file.write(mlr.summary(alpha=.05).as_text())  # printing the summary to a file, ALPHA is set to .05
    distribution_plot(y, yhat, response_var)


"""

Loading the data into Pandas Dataframe
Using Pandas function get the categorical features
read for use
Categorical features used: RAD, CHAS
"""


def boston_housing():
    df = pd.read_csv("BostonHousingDataset/BostonHousingData.csv")  # loading the csv file
    dummies = pd.get_dummies(df["RAD"], drop_first=True)  # removing the first level reduces multicollinearity
    df = df.join(dummies)  # merging the dataframes
    df = df.drop(columns=["RAD"])  # dropping the original categorical variable columns
    dummies = pd.get_dummies(df["CHAS"], drop_first=True)
    df = df.join(dummies)
    df = df.drop(columns=["CHAS"])
    x_names = ["CRIM", "RM", "PTRATIO", "LSTAT", "NOX", "TAX", 1, 2, 3, 4, 5, 6, 7, 8, 24]  # independent vars list
    response_var = "MEDV"
    y = df[[response_var]]
    x = df[x_names]
    file_path = "BostonHousingDataset/summary.txt"
    multiple_lr(x, y, "Predicted MEDV", file_path, response_var)


def grid_stability():
    df = pd.read_csv("Chosen/Data_for_UCI_named_1.csv")
    response_var = ["stabf"]  # the response var
    predictor_var = ["tau1", "p1", "p2", "p3", "g1", "g2", "g3", "g4"]  # predictor var

    x = df[predictor_var]
    y = df[response_var]
    x = sm.add_constant(x)
    # splitting the data into train(90%) and test(10%)
    # random state is set to a integer to randomly sample the data for test and training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=42)

    x_train = sm.add_constant(x_train)
    log_reg = sm.Logit(y_train, x_train).fit()
    # print(x_train)

    yhat = log_reg.predict(x_test)
    prediction = list(map(round, yhat))
    cm = confusion_matrix(y_test, prediction)
    print("Confusion matrix: \n", cm)
    # print("Test accuracy: ", accuracy_score(y_test, prediction))
    # print(log_reg.summary(0.5))
    # count_plot(response_var[0], df)
    print(classification_report(y_test, prediction))


# boston_housing()
grid_stability()
