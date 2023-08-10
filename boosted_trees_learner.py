import os
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import warnings
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from statsmodels.tools import add_constant
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from loguru import logger
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate

from data_preparation import exploratory_analysis

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.max_columns", None)

Admissions = pd.read_excel("IPEDS_data.xlsx")

# Filtering dataset for input and output variables only

AdmissionsSlim = Admissions.filter(
    [
        "Percent admitted - total",
        "ACT Composite 75th percentile score",
        "Historically Black College or University",
        "Total  enrollment",
        "Total price for out-of-state students living on campus 2013-14",
        "Percent of total enrollment that are White",
        "Percent of total enrollment that are women",
    ]
).dropna()

AdmissionsSlim.columns

AdmissionsSlim.columns = [
    "Per_Admit",
    "ACT_75TH",
    "Hist_Black",
    "Total_ENROLL",
    "Total_Price",
    "Per_White",
    "Per_Women",
]

# Defining 'Selective' as an Admittance Rate Under 50%
AdmissionsSlim["Per_Admit"] = np.where(AdmissionsSlim["Per_Admit"] < 50, 1, 0)
AdmissionsSlim["Hist_Black"] = np.where(
    AdmissionsSlim["Hist_Black"] == "Yes", 1, 0
)

# Create a new variable, which is the percentage of total enrollment that are non-white.
AdmissionsSlim = AdmissionsSlim.assign(
    Per_Non_White=lambda a: 100 - a.Per_White
)

exploratory_analysis(AdmissionsSlim)

X = AdmissionsSlim[
    [
        "ACT_75TH",
        "Hist_Black",
        "Total_ENROLL",
        "Total_Price",
        "Per_Non_White",
        "Per_Women",
    ]
]

y = AdmissionsSlim[["Per_Admit"]]


# Split dataframe into training and testing data. Remember to set a seed.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=47
)


def scale_and_add_coefficient_to_independent_variables(pipeline, X_dataframe):
    """

    :param pipeline:
    :param X_dataframe:
    :return: X_dataframe_scaled
    """
    X_dataframe_scaled = pipeline.named_steps["scaler"].transform(X_dataframe)
    X_dataframe_scaled = add_constant(X_dataframe_scaled)
    return X_dataframe_scaled


# The StandardScaler is only being applied to the independent vars here.

steps = [("scaler", StandardScaler()), ("model", LinearRegression())]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

X_train_scaled = scale_and_add_coefficient_to_independent_variables(
    pipeline, X_train
)

model = sm.OLS(y_train, X_train_scaled)
results = model.fit()

print(results.summary(xname=["intercept"] + list(X_train.columns)))

X_test_scaled = scale_and_add_coefficient_to_independent_variables(
    pipeline, X_test
)

y_pred = pipeline.predict(X_test)

mae = round(mean_absolute_error(y_test, y_pred), 2)

print("MAE:", mae)

steps = [
    ("scaler", StandardScaler()),
    ("model", LGBMRegressor()),
]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

X_train_scaled = scale_and_add_coefficient_to_independent_variables(
    pipeline, X_train
)

kfold = KFold(n_splits=10, shuffle=True, random_state=47)

n_scores = cross_val_score(
    pipeline,
    X_train_scaled,
    y_train,
    scoring="neg_mean_squared_error",
    cv=kfold,
    n_jobs=-1,
    error_score="raise",
)

print("Accuracy: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))

pipeline.fit(X_test, y_test)

X_test_scaled = scale_and_add_coefficient_to_independent_variables(
    pipeline, X_test
)

kfold = KFold(n_splits=10, shuffle=True, random_state=47)

n_scores = cross_val_score(
    pipeline,
    X_test_scaled,
    y_test,
    scoring="neg_mean_squared_error",
    cv=kfold,
    n_jobs=-1,
    error_score="raise",
)

print("MAE: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))

coefficient_list = X_train.columns
coefficient_list = list(coefficient_list)
print(coefficient_list)


def create_a_test_array(
        ACT_75TH, Hist_Black, Total_ENROLL, Total_Price, Per_Non_White, Per_Women
):
    row = [
        [ACT_75TH, Hist_Black, Total_ENROLL, Total_Price, Per_Non_White, Per_Women]
    ]

    return row


row = create_a_test_array(
    ACT_75TH=35, Hist_Black=1, Total_ENROLL=20000, Total_Price=70000, Per_Non_White=10, Per_Women=60
)

yhat = pipeline.predict(row)
print("Prediction: %.3f" % yhat[0])
