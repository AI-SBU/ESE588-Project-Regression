import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor  # dealing with multicollinearity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
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
    # plt.show()


def count_plot(column_name, dataframe, model_name, file_path):
    sns.countplot(x=column_name, data=dataframe)
    plt.title("Outcome: Positive vs Negative")
    plt.savefig(file_path + model_name + "_count_plot.png")
    # plt.show()


def heat_map(confusion, model_name, file_path):  # to visualize the confusion matrix
    sns.heatmap(confusion, annot=True, fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(file_path + model_name + "_cm_heat_map.png")
    # plt.show()


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


def multiple_lr(x, y, file_path, response_var):
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
    multiple_lr(x, y, file_path, response_var)


"""
The function below performs up-sampling on the Grid-Stability dataset
returns the newly upsampled dataset
"""


def grid_stability_upsampled_model(df):
    # Separate
    df_majority = df[df.stabf == 0]
    df_minority = df[df.stabf == 1]

    majority_counts = len(df_majority)  # 6,380 counts of 0

    # Upsampling
    df_minority_upsampled = resample(df_minority,  # data to up-sample
                                     replace=True,  # resamples with replacement
                                     n_samples=majority_counts,  # total samples needed to match the majority count
                                     random_state=42)  # random seed

    #  Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled


def grid_stability_downsampled_model(df):
    df_majority = df[df.stabf == 0]  # new dataframe containing majority class
    df_minority = df[df.stabf == 1]  # new dataframe containing the minority class

    minority_counts = len(df_minority)  # 10,000 - 6,380 counts of 0

    df_majority_downsampled = resample(df_majority,  # data to down-sample
                                       replace=False,
                                       n_samples=minority_counts,  # to match the minority count
                                       random_state=42)  # random seed

    df_downsampled = pd.concat([df_majority_downsampled, df_minority])  # combining the new majority with minority

    print(df_downsampled.stabf.value_counts())
    return df_downsampled


def grid_stability_random_forest_classifier_model():
    df = pd.read_csv("Chosen/Data_for_UCI_named_1.csv")
    response_var = ["stabf"]  # the response var
    predictor_var = ["tau1", "p1", "p2", "p3", "g1", "g2", "g3", "g4"]  # predictor var
    model_name = "RandomForestClassifier"
    file_path = "Chosen/modelThree_randomForest/"
    x = df[predictor_var]
    y = df[response_var]
    # x = sm.add_constant(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, np.ravel(y_train))
    yhat = rfc.predict(x_test)
    prediction = list(map(round, yhat))
    print(np.unique(prediction))
    print(accuracy_score(y_test, prediction))


"""
The function below builds a logistic regression model on the Electric Grid Stability data set
It uses 8 predictor variable to predict 1 binary response variable
All of the predictor variable are numeric (not categorical)
Change the "file_path" var to a destination of your choice
The summary/reports/plots will all be saved in this destination
Leaving it empty will save the files on the project folder
"""


def grid_stability():
    df = pd.read_csv("Chosen/Data_for_UCI_named_1.csv")
    response_var = ["stabf"]  # the response var
    predictor_var = ["tau1", "p1", "p2", "p3", "g1", "g2", "g3", "g4"]  # predictor var
    model_name = "Downsampled"
    file_path = "Chosen/modelThree_downsampled/"

    # df = grid_stability_upsampled_model(df)
    df = grid_stability_downsampled_model(df)

    x = df[predictor_var]
    y = df[response_var]
    x = sm.add_constant(x)  # y-intercept

    # splitting the data into train(80%) and test(20%)
    # random state is set to a integer to randomly sample the data for test and training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    log_reg = sm.Logit(y_train, x_train).fit()  # fit the model
    yhat = log_reg.predict(x_test)  # make predictions using the test features

    # print(np.unique(yhat))
    prediction = list(map(round, yhat))  # rounding to 1 or 0, yhat prior to this has floating point values
    # threshold is 0.5

    with open(file_path + model_name + "_classification_report.txt", "w") as file:
        file.write(classification_report(y_test, prediction))
        file.write("\nTest Accuracy: " + str(accuracy_score(y_test, prediction)))
        file.write("\n")
        file.close()

    with open(file_path + model_name + "_summary.txt", "w") as file:
        file.write(log_reg.summary(0.5).as_text())
        file.close()
    # print(log_reg.summary(0.5))
    count_plot(response_var[0], df, model_name, file_path)  # visualizing the response var elements using a countplot
    heat_map(confusion_matrix(y_test, prediction), model_name,
             file_path)  # visualizing confusion matrix using a heatmap


# boston_housing()
grid_stability()
# grid_stability_random_forest_classifier_model()
