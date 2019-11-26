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
from evaluate import *


train_data = pd.read_csv('../../data/Kmers6_counts_600bp.csv')
train_methys = pd.read_csv('../../data/Mouse_DMRs_methylation_level.csv',header = None)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 1)

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

cell_type = 5
X = train_data.as_matrix()
X = np.concatenate([X,np.ones((len(X),1))],axis = 1)
train_X = X[train_indices]
val_X = X[val_indices]

multiclass = True
if multiclass:
    Y = np.array(train_methys)
else:
    Y = np.array(train_methys[cell_type])

train_Y = Y[train_indices]
val_Y = Y[val_indices]

epochs = 200

net = Net(X.shape[1],Y.shape[1]).to(computing_device)
net.apply(weights_init)

criterion = nn.MSELoss()

#Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(net.parameters(),lr = 0.0001,weight_decay=0.00001)

print(train_X.shape,train_Y.shape,val_X.shape,val_Y.shape)

for e in range(epochs):

    # Train data

    train_loss = 0
    batch_count = 0
    for i in range(len(train_X)//batch_size+1):
        if i*batch_size >= len(train_X):continue
        x = torch.tensor(train_X[i*batch_size:(i+1)*batch_size]).to(computing_device)
        y = torch.tensor(train_Y[i*batch_size:(i+1)*batch_size]).view(-1,1).to(computing_device)
        optimizer.zero_grad()

        outputs = net(x.float())
        #print(x.shape,outputs.shape,y.shape)
        #train_pred.append(outputs.item())
        #print(outputs.shape,y.float().shape)
        loss = torch.sqrt(criterion(outputs,y.float()))
        #regularization_loss = 0
        #for param in net.parameters():
        #    regularization_loss += torch.sum(torch.abs(param))
        #loss += regularization_loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        batch_count += 1
    
    #train_result = evaluate(net,train_X,train_Y,computing_device)
    print('Epoch {}, Train Batch Loss: {}, '.format(e,train_loss/batch_count))
    # Validate data
    val_loss = 0
    val_pred = []
    batch_count = 0
    for i in range(len(val_X)//batch_size+1):

        with torch.no_grad():
            x = torch.tensor(val_X[i*batch_size:(i+1)*batch_size]).to(computing_device)
            y = torch.tensor(val_Y[i*batch_size:(i+1)*batch_size]).view(-1,1).to(computing_device)
            outputs = net(x.float())
            loss = torch.sqrt(criterion(outputs,y.float()))
            val_loss += loss.item()
            batch_count += 1
    
    if not multiclass:
        val_result = evaluate(net,val_X,val_Y,computing_device)
        print('\rEpoch {}, Val Loss: {}, Val R2 Score:{}'.format(e,val_loss/batch_count, val_result[1]))
    else:
        evaluateMultiClass(net,val_X,val_Y,computing_device)






