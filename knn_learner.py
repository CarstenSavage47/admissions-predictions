from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas
import numpy


def knn_learner(x_train, y_train, x_test, y_test):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    """
    accuracy_values = []

    for i in range(1, 100):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)

        # Calculate the accuracy of the model
        if i % 2 == 0:
            print(
                "Iteration K =",
                i,
                "Accuracy Rate=",
                knn.score(x_test, y_test),
            )
            print(knn.score(x_test, y_test))
            accuracy_values.append(
                [i, knn.score(x_test, y_test)]
            )

    k_accuracy_pair = pandas.DataFrame(accuracy_values)
    k_accuracy_pair.columns = ["K", "Accuracy"]

    # Let's see the K value where the accuracy was best:

    k_accuracy_pair[
        k_accuracy_pair["Accuracy"]
        == max(k_accuracy_pair["Accuracy"])
    ]

    # Best iteration was K = 41 and K = 47 and K = 51, all three with Accuracy = 89.3%.
    # This is actually slightly better than the neural network's accuracy.
    # The neural network's accuracy was 87.23%.

