
# Thank you to Venelin (https://curiousily.com/about) /
# (https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/)
# And to StatQuest (https://www.youtube.com/watch?v=FHdlXe1bSe4)
# For help with getting started with this neural network.

# The purpose of this neural network is to predict whether a college is selective or not based on attributes.
# The current model is using 75th percentile ACT scores, admittance rate percentages,
# ... total enrollment, total out-of-state price, percent of enrollment that is non-white,
# ... and percent of total enrollment that is women as predictors for whether a college is selective.
# I defined a 'selective' university as one that has an acceptance rate of less than 65%.

import torch  # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network.
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD  # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
import matplotlib.pyplot as plt  ## matplotlib allows us to draw graphs.
import seaborn as sns  ## seaborn makes it easier to draw nice-looking graphs.
import os
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing

import pandas
import numpy as np

Admissions = pandas.read_excel('/Users/carstenjuliansavage/Desktop/IPEDS_data.xlsx')
pandas.set_option('display.max_columns', None)

# Filtering dataset for input and output variables only

AdmissionsSlim = (Admissions
    .filter(['Percent admitted - total',
             'ACT Composite 75th percentile score',
             'Historically Black College or University',
             'Total  enrollment',
             'Total price for out-of-state students living on campus 2013-14',
             'Percent of total enrollment that are White',
             'Percent of total enrollment that are women'])
    .dropna()
)

AdmissionsSlim.columns

AdmissionsSlim.columns = ['Per_Admit','ACT_75TH','Hist_Black','Total_ENROLL','Total_Price','Per_White','Per_Women']

# Defining 'Selective' as an Admittance Rate Under 65%
AdmissionsSlim['Per_Admit'] = np.where(AdmissionsSlim['Per_Admit'] < 65,1,0)
AdmissionsSlim['Hist_Black'] = np.where(AdmissionsSlim['Hist_Black'] == 'Yes',1,0)

# Create a new variable, which is the percentage of total enrollment that are non-white.
AdmissionsSlim = (AdmissionsSlim
    .assign(Per_Non_White=lambda a: 100-a.Per_White)
)

X = AdmissionsSlim[['ACT_75TH',
                    'Hist_Black',
                    'Total_ENROLL',
                    'Total_Price',
                    'Per_Non_White',
                    'Per_Women']]
y = AdmissionsSlim[['Per_Admit']]

# Split dataframe into training and testing data. Remember to set a seed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# Scaling the data to be between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

# Turning the training and testing datasets into tensors
X_train = torch.tensor(X_train)
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
X_test = torch.tensor(X_test)
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())
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
    self.fc1 = nn.Linear(n_features, 12)
    self.fc2 = nn.Linear(12, 8)
    self.fc3 = nn.Linear(8, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))
net = Net(X_train.shape[1])

# Loss Function
criterion = nn.BCELoss()
optimizer = SGD(net.parameters(), lr=000.1)  ## here we're creating an optimizer to train the neural network.
#This learning rate seems to be working well so far

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)
net = net.to(device)
criterion = criterion.to(device)

def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
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

    print(f'''    Epoch {epoch}
    Training loss: {round_tensor(train_loss)} Accuracy: {round_tensor(train_acc)}
    Testing loss: {round_tensor(test_loss)} Accuracy: {round_tensor(test_acc)}''')

# If test loss is less than 0.02, then break. That result is satisfactory.
    if test_loss < 0.02:
        print("Num steps: " + str(epoch))
        break

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


# Creating a function to evaluate our input
def AreWeSelective(ACT_75TH,
                   Hist_Black,
                   Total_ENROLL,
                   Total_Price,
                   Per_Non_White,
                   Per_Women
                   ):
  t = torch.as_tensor([ACT_75TH,
                       Hist_Black,
                       Total_ENROLL,
                       Total_Price,
                       Per_Non_White,
                       Per_Women
                       ]) \
    .float() \
    .to(device)
  output = net(t)
  return output.ge(0.5).item()


# Input values between 0 and 1 to test (since it's scaled).
# For example, Total_Price = 1.0 signifies the highest price in the dataset.
AreWeSelective(ACT_75TH=0.1,
               Hist_Black=0.1,
               Total_ENROLL=0.1,
               Total_Price=0.1,
               Per_Non_White=0.1,
               Per_Women=0.1
               )

AreWeSelective(ACT_75TH=0.9,
               Hist_Black=0.5,
               Total_ENROLL=0.5,
               Total_Price=0.5,
               Per_Non_White=0.5,
               Per_Women=0.9
               )

# It looks like 75th percentile ACT scores are a good predictor for whether or not a school is
# ... selective based on the criteria. In the model, 75th percentile ACT scores are assigned the greatest weight.