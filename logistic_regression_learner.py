import pandas
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures


def logistic_regression_learner(
    x_train, y_train, x_test, y_test
):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    """
    Logit = LogisticRegression()

    poly_accuracy = []

    polynomials = range(1, 10)

    for poly_degree in polynomials:
        poly = PolynomialFeatures(
            degree=poly_degree, include_bias=False
        )
        X_poly = poly.fit_transform(x_train)
        x_test_poly = poly.fit_transform(x_test)
        Logit.fit(X_poly, y_train)
        y_pred = Logit.predict(x_test_poly)
        print(
            "Polynomial Degree:",
            poly_degree,
            "Accuracy:",
            round(Logit.score(x_test_poly, y_test), 3),
        )
        poly_accuracy.append(
            [
                poly_degree,
                round(Logit.score(x_test_poly, y_test), 3),
            ]
        )

    Polynomial_Accuracy = pandas.DataFrame(poly_accuracy)
    Polynomial_Accuracy.columns = ["Polynomial", "Accuracy"]

    from sklearn.metrics import confusion_matrix

    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
