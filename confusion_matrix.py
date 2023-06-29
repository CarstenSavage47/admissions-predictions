import seaborn as sns  ## seaborn makes it easier to draw nice-looking graphs.
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
)

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
