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

from knn_learner import knn_learner


def make_confusion_matrix(y_test, y_test_pred):
    """

    :param y_test:
    :param y_test_pred:
    """
    # Define categories for our confusion matrix
    categories = ["Not Selective", "Selective"]

    # Where y_test_pred > 0.5, we categorize it as 1, or else 0.
    y_test_dummy = np.where(y_test_pred > 0.5, 1, 0)

    # Creating a confusion matrix to visualize the results.
    # Model Evaluation Part 2
    a_confusion_matrix = confusion_matrix(
        y_test, y_test_dummy
    )
    confusion_frame = pandas.DataFrame(
        a_confusion_matrix,
        index=categories,
        columns=categories,
    )
    sns.heatmap(confusion_frame, annot=True, fmt="g")
    plt.ylabel("Observed")
    plt.xlabel("Yhat")
