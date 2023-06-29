# Thank you to Venelin (https://curiousily.com/about) /
# (https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/)
# And to StatQuest (https://www.youtube.com/watch?v=FHdlXe1bSe4)
# For help with getting started with this neural network.

# The purpose of this neural network is to predict whether a college is selective or not based on attributes.
# The current model is using 75th percentile ACT scores, admittance rate percentages,
# ... total enrollment, total out-of-state price, percent of enrollment that is non-white,
# ... historically black university status (dummy variable),
# ... and percent of total enrollment that is women as predictors for whether a college is selective.
# I defined a 'selective' university as one that has an acceptance rate of less than 50%.

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

Admissions = pandas.read_excel("IPEDS_data.xlsx")
pandas.set_option("display.max_columns", None)

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
AdmissionsSlim["Per_Admit"] = np.where(
    AdmissionsSlim["Per_Admit"] < 50, 1, 0
)
AdmissionsSlim["Hist_Black"] = np.where(
    AdmissionsSlim["Hist_Black"] == "Yes", 1, 0
)

# Create a new variable, which is the percentage of total enrollment that are non-white.
AdmissionsSlim = AdmissionsSlim.assign(
    Per_Non_White=lambda a: 100 - a.Per_White
)

## Exploratory Data Analysis

corrMatrix = AdmissionsSlim.corr()
print(corrMatrix)

# Correlation Matrix Version 1
fig = px.imshow(corrMatrix)

## Correlation Matrix Version 2

Corr_Heat = sns.heatmap(
    corrMatrix,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 10},
    yticklabels=corrMatrix.columns,
    xticklabels=corrMatrix.columns,
    cmap="Spectral_r",
)
plt.show()

Pair_Plot = px.scatter_matrix(AdmissionsSlim)
Pair_Plot.show()

# Evaluating kurtosis:
# 1. Mesokurtic: Data follows a normal distribution
# 2. Leptokurtic: Heavy tails on either side, indicating large outliers. Looks like Top-Thrill Dragster.
# 3. Playtkurtic: Flat tails indicate that there aren't many outliers.

# A kurtosis value greater than +1 indicates the graph is very peaked. Leptokurtic.
# A kurtosis value less than -1 indicates the graph is relatively flat. Playtkurtic.
# A kurtosis value of 0 indicates that the graph follows a normal distribution. Mesokurtic.

# Evaluating skewness:
# 1. A negative value indicates the tail is on the left side of the distribution.
# 2. A positive value indicates the tail is on the right side of the distribution.
# 3. A value of zero indicates that there is no skewness in the distribution; it's perfectly symmetrical.

Kurtosis_x_Skewness = []

for col in AdmissionsSlim:
    print(
        f"Skewness for {col}: {AdmissionsSlim[col].skew()}"
    )
    print(
        f"Kurtosis for {col}: {AdmissionsSlim[col].kurt()}"
    )
    Kurtosis_x_Skewness.append(
        {
            "Column": col,
            "Skewness": AdmissionsSlim[col].skew(),
            "Kurtosis": AdmissionsSlim[col].kurt(),
        }
    )

Kurtosis_x_Skewness_DF = pandas.DataFrame(
    Kurtosis_x_Skewness
)

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
    X, y, test_size=0.2, random_state=47, stratify=y
)

# Scaling the data to be between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

# Let's confirm that the scaling worked as intended.
# All values should be between 0 and 1 for all variables.
X_Stats = pandas.DataFrame(X_train)
pandas.set_option("display.max_columns", None)
X_Stats.describe()

y_train_Stats = pandas.DataFrame(y_train)
y_test_Stats = pandas.DataFrame(y_test)
y_train_Stats.describe()
y_test_Stats.describe()

# We can see that the data has stratified as intended.

# Turning the training and testing datasets into tensors
X_train = torch.tensor(X_train)
y_train = torch.squeeze(
    torch.from_numpy(y_train.to_numpy()).float()
)
X_test = torch.tensor(X_test)
y_test = torch.squeeze(
    torch.from_numpy(y_test.to_numpy()).float()
)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

X_train = X_train.float()
y_train = y_train.float()
X_test = X_test.float()
y_test = y_test.float()


# Initializing the neural network class
class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    # It seems that it has helped to increase the number of hidden layers and nodes per layer.

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


net = Net(X_train.shape[1])

# Loss Function
criterion = nn.BCELoss()
optimizer = SGD(
    net.parameters(), lr=1.0
)  ## here we're creating an optimizer to train the neural network.
# This learning rate seems to be working well so far

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)
net = net.to(device)
criterion = criterion.to(device)


def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(0.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


for epoch in range(1000):
    y_pred = net(X_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)
    train_acc = calculate_accuracy(y_train, y_pred)
    y_test_pred = net(X_test)
    y_test_pred = torch.squeeze(y_test_pred)
    test_loss = criterion(y_test_pred, y_test)
    test_acc = calculate_accuracy(y_test, y_test_pred)

    print(
        f"""    Epoch {epoch}
    Training loss: {round_tensor(train_loss)} Accuracy: {round_tensor(train_acc)}
    Testing loss: {round_tensor(test_loss)} Accuracy: {round_tensor(test_acc)}"""
    )

    # If test loss is less than 0.02, then break. That result is satisfactory.
    if test_loss < 0.02:
        print("Num steps: " + str(epoch))
        break

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


# Creating a function to evaluate our input
def AreWeSelective(
    ACT_75TH,
    Hist_Black,
    Total_ENROLL,
    Total_Price,
    Per_Non_White,
    Per_Women,
):
    t = (
        torch.as_tensor(
            [
                ACT_75TH,
                Hist_Black,
                Total_ENROLL,
                Total_Price,
                Per_Non_White,
                Per_Women,
            ]
        )
        .float()
        .to(device)
    )
    output = net(t)
    return output.ge(0.5).item(), output.item()


# Try experimenting with the following functions and evaluate classification output.
# Input values between 0 and 1 to test (since it's scaled).
# For example, Total_Price = 1.0 signifies the highest price in the dataset.
AreWeSelective(
    ACT_75TH=0.1,
    Hist_Black=0.1,
    Total_ENROLL=0.1,
    Total_Price=0.1,
    Per_Non_White=0.1,
    Per_Women=0.1,
)

AreWeSelective(
    ACT_75TH=0.9,
    Hist_Black=0.5,
    Total_ENROLL=0.9,
    Total_Price=0.9,
    Per_Non_White=0.9,
    Per_Women=0.9,
)

# It looks like 75th percentile ACT scores are a good predictor for whether or not a school is
# ... selective based on the criteria. In the model, 75th percentile ACT scores are assigned the greatest weight.

# Preparation for confusion matrix

# Define categories for our confusion matrix
Categories = ["Not Selective", "Selective"]

# Where y_test_pred > 0.5, we categorize it as 1, or else 0.
y_test_dummy = np.where(y_test_pred > 0.5, 1, 0)

# Creating a confusion matrix to visualize the results.
# Model Evaluation Part 2
Confusion_Matrix = confusion_matrix(y_test, y_test_dummy)
Confusion_DF = pandas.DataFrame(
    Confusion_Matrix, index=Categories, columns=Categories
)
sns.heatmap(Confusion_DF, annot=True, fmt="g")
plt.ylabel("Observed")
plt.xlabel("Yhat")

# Let's conduct a linear regression and evaluate the coefficients.

Reg_Out = ols(
    "Per_Admit ~ ACT_75TH + Hist_Black + Total_ENROLL + Total_Price + Per_Non_White + Per_Women",
    data=AdmissionsSlim,
).fit()

print(Reg_Out.summary())


# Example regression table. We can see that increases in ACT_75TH, HIST_Black, and Per_Non_White have
# The largest effects on PER_ADMIT.

#                      coef      std err       t        P>|t|      [0.025      0.975]
#   ---------------------------------------------------------------------------------
#   Intercept        -0.8614      0.161     -5.342      0.000      -1.178      -0.545
#   ACT_75TH          0.0445      0.006      6.976      0.000       0.032       0.057
#   Hist_Black        0.0650      0.086      0.754      0.451      -0.104       0.234
#   Total_ENROLL  -2.902e-06   1.52e-06     -1.909      0.057   -5.88e-06     8.1e-08
#   Total_Price    -2.01e-06   1.91e-06     -1.052      0.293   -5.76e-06    1.74e-06
#   Per_Non_White     0.0087      0.001      9.674      0.000       0.007       0.011
#   Per_Women        -0.0009      0.001     -0.721      0.471      -0.003       0.002

# Note: The probability value on Hist_Black indicates that if the null hypothesis is true, and Hist_Black has no effect
# ...on Per_Admit, the likelihood of getting the results we did is 45.1%.


# Revisiting the functions: When ACT_75TH, Hist_Black, and Per_Non_White = 1.0,
# ...the bound of the scaled data, function AreWeSelective outputs True.

AreWeSelective(
    ACT_75TH=1.0,
    Hist_Black=1.0,
    Total_ENROLL=0.0,
    Total_Price=0.0,
    Per_Non_White=1.0,
    Per_Women=0.0,
)

AreWeSelective(
    ACT_75TH=0.2,
    Hist_Black=0.2,
    Total_ENROLL=0.2,
    Total_Price=0.2,
    Per_Non_White=0.5,
    Per_Women=0.5,
)


# Let's create a quick K-nearest neighbors model and see what we get.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas
import numpy

Accuracy_Values = []

for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

    # Calculate the accuracy of the model
    if i % 2 == 0:
        print(
            "Iteration K =",
            i,
            "Accuracy Rate=",
            knn.score(X_test, y_test),
        )
        print(knn.score(X_test, y_test))
        Accuracy_Values.append(
            [i, knn.score(X_test, y_test)]
        )

K_Accuracy_Pair = pandas.DataFrame(Accuracy_Values)
K_Accuracy_Pair.columns = ["K", "Accuracy"]

# Let's see the K value where the accuracy was best:

K_Accuracy_Pair[
    K_Accuracy_Pair["Accuracy"]
    == max(K_Accuracy_Pair["Accuracy"])
]

# Best iteration was K = 41 and K = 47 and K = 51, all three with Accuracy = 89.3%.
# This is actually slightly better than the neural network's accuracy.
# The neural network's accuracy was 87.23%.


# Let's try comparing these results to a logistic regression model.

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

Logit = LogisticRegression()

poly_accuracy = []

polynomials = range(1, 10)

for poly_degree in polynomials:
    poly = PolynomialFeatures(
        degree=poly_degree,
        interaction_only=False,
        include_bias=False,
    )
    X_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    Logit.fit(X_poly, y_train)
    y_pred = Logit.predict(X_test_poly)
    print(
        "Polynomial Degree:",
        poly_degree,
        "Accuracy:",
        round(Logit.score(X_test_poly, y_test), 3),
    )
    poly_accuracy.append(
        [
            poly_degree,
            round(Logit.score(X_test_poly, y_test), 3),
        ]
    )

Polynomial_Accuracy = pandas.DataFrame(poly_accuracy)
Polynomial_Accuracy.columns = ["Polynomial", "Accuracy"]

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# We can see that the optimal polynomial degree is 9.
# Our k-nearest neighbors model was the most accurate, with accuracy of 89.3%.
# The neural network's accuracy was 87.23%.
# In contrast, our logistic regression model has an accuracy of 88.9%.
# My XGBoost model (which is also published on Github) also has an accuracy rate of 87.23% for this dataset.
