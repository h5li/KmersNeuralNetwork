import pandas as pd
import numpy as np
import torch
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys

from model import *


train_data = pd.read_csv('../data/Kmers6_counts_600bp.csv')
train_methys = pd.read_csv('../data/Mouse_DMRs_counts_methylated.csv',header = None)


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 8, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = {}
    print("CUDA NOT supported")

batch_size = 64
validation_split = 0.2
shuffle_dataset = True
random_seed= 24

dataset_size = len(train_data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

epochs = 50

net = Net(train_data.shape[1],1).to(computing_device)
criterion = nn.MSELoss()

#Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(net.parameters(),lr = 0.001)

cell_type = 5
X = train_data.as_matrix()
train_X = X[train_indices]
val_X = X[val_indices]
Y = train_methys[cell_type]

train_Y = Y[train_indices]
val_Y = Y[val_indices]

for e in range(epochs):

    # Train data
    optimizer.zero_grad()

    train_loss = 0
    for i,x in enumerate(train_X):
        x = torch.tensor(x).to(computing_device)
        y = torch.tensor(train_Y[i]).to(computing_device)

        outputs = net(x)
        loss = criterion(outputs,y)
        train_loss += loss.item()
        loss.backward()
    
    optimizer.step()

    sys.stdout.write('\rEpoch {}, Train Loss: {}'.format(e,train_loss/len(train_X)))
    print("\n")

    # Validate data
    val_loss = 0
    for i,x in enumerate(val_X):

        with torch.no_grad():
            x = torch.tensor(x).to(computing_device)
            y = torch.tensor(val_Y[i]).to(computing_device)

            outputs = net(x)
            loss = criterion(outputs,y)
            val_loss += loss.item()
    sys.stdout.write('\rEpoch {}, Val Loss: {}'.format(e,val_loss/len(val_X)))
    print("\n")






