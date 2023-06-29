import torch  # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network.
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import (
    SGD,
)  # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
from statsmodels.formula.api import ols
import pandas
import numpy as np


from data_preparation import (
    exploratory_analysis,
    train_test_splitter,
)
from knn_learner import knn_learner
from confusion_matrix import make_confusion_matrix


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

x_train, y_train, x_test, y_test = train_test_splitter(X, y)

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


net = Net(x_train.shape[1])

# Loss Function
criterion = nn.BCELoss()
optimizer = SGD(
    net.parameters(), lr=1.0
)  ## here we're creating an optimizer to train the neural network.
# This learning rate seems to be working well so far

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
x_train = x_train.to(device)
y_train = y_train.to(device)

x_test = x_test.to(device)
y_test = y_test.to(device)
net = net.to(device)
criterion = criterion.to(device)


def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(0.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


for epoch in range(1000):
    y_pred = net(x_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)
    train_acc = calculate_accuracy(y_train, y_pred)
    y_test_pred = net(x_test)
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


make_confusion_matrix(y_test, y_test_pred)


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


AreWeSelective(
    ACT_75TH=0.1,
    Hist_Black=0.1,
    Total_ENROLL=0.1,
    Total_Price=0.1,
    Per_Non_White=0.1,
    Per_Women=0.1,
)

knn_learner(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
)

# Linear regression

linear_regression_output = ols(
    "Per_Admit ~ ACT_75TH + Hist_Black + Total_ENROLL + Total_Price + Per_Non_White + Per_Women",
    data=AdmissionsSlim,
).fit()

print(linear_regression_output.summary())
