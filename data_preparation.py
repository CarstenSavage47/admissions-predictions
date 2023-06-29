import torch  # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network.
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import (
    SGD,
)  # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
import matplotlib.pyplot as plt  ## matplotlib allows us to draw graphs.
import seaborn as sns  ## seaborn makes it easier to draw nice-looking graphs.
import os
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
from sklearn import preprocessing
from statsmodels.formula.api import ols
import plotly.express as px

import pandas
import numpy as np


def exploratory_analysis(master_data):
    """

    :param master_data:
    """
    corr_matrix = master_data.corr()
    print(corr_matrix)

    fig = px.imshow(corr_matrix)

    corr_heat = sns.heatmap(
        corr_matrix,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        yticklabels=corr_matrix.columns,
        xticklabels=corr_matrix.columns,
        cmap="Spectral_r",
    )
    plt.show()

    pair_plot = px.scatter_matrix(master_data)
    pair_plot.show()

    kurtosis_x_skewness = []

    for col in master_data:
        print(
            f"Skewness for {col}: {master_data[col].skew()}"
        )
        print(
            f"Kurtosis for {col}: {master_data[col].kurt()}"
        )
        kurtosis_x_skewness.append(
            {
                "Column": col,
                "Skewness": master_data[col].skew(),
                "Kurtosis": master_data[col].kurt(),
            }
        )

    kurtosis_x_skewness_df = pandas.DataFrame(
        kurtosis_x_skewness
    )


def train_test_splitter(x, y):
    """

    :param x:
    :param y:
    :return: X_train, y_train, X_test, y_test
    """
    # Split dataframe into training and testing data. Remember to set a seed.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=47, stratify=y
    )

    # Scaling the data to be between 0 and 1
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)

    # Let's confirm that the scaling worked as intended.
    # All values should be between 0 and 1 for all variables.
    x_stats = pandas.DataFrame(x_train)
    pandas.set_option("display.max_columns", None)
    x_stats.describe()

    y_train_stats = pandas.DataFrame(y_train)
    y_test_Stats = pandas.DataFrame(y_test)
    y_train_stats.describe()
    y_test_Stats.describe()

    # We can see that the data has stratified as intended.

    # Turning the training and testing datasets into tensors
    x_train = torch.tensor(x_train)
    y_train = torch.squeeze(
        torch.from_numpy(y_train.to_numpy()).float()
    )
    x_test = torch.tensor(x_test)
    y_test = torch.squeeze(
        torch.from_numpy(y_test.to_numpy()).float()
    )
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    x_train = x_train.float()
    y_train = y_train.float()
    x_test = x_test.float()
    y_test = y_test.float()

    return x_train, y_train, x_test, y_test
