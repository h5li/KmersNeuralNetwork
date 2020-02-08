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


train_data = np.load("../DNA_seq.npy")[:,200:800,:]
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

def convertMotifToVector(motif):
    matrix = []
    for c in motif:
        if c == 'A':
            matrix.append([1,0,0,0])
        elif c == 'C':
            matrix.append([0,1,0,0])
        elif c == 'G':
            matrix.append([0,0,1,0])
        else:
            matrix.append([0,0,0,1])
    #print(np.array(sequence_vector).shape)
    return np.array(matrix).T

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
random_seed= 6

dataset_size = len(train_data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

cell_type = 5
X = train_data.reshape(len(train_data),4,600)
#X = np.concatenate([X,np.ones((len(X),1))],axis = 1)
train_X = X[train_indices]
val_X = X[val_indices]

multiclass = False
if multiclass:
    Y = np.array(train_methys)
else:
    Y = np.array(train_methys[cell_type])

train_Y = Y[train_indices]
val_Y = Y[val_indices]

epochs = 500

num_features_selected = 32
filter_size = 6

net = CNNnet(num_features_selected,filter_size).to(computing_device)
net.apply(weights_init)

pretrained = True
if pretrained:
    features = np.load('LASSO_SelectedFeatures.npy')[:num_features_selected,2]
    matrices = []
    for f in features:
        matrices.append(convertMotifToVector(f))
    matrices = np.array(matrices)
    with torch.no_grad():
        net.main[0].weight.data = torch.Tensor(matrices).to(computing_device)

print(net)

#criterion = nn.MSELoss()

def criterion(y,yhat,weights):
    loss = nn.MSELoss()(y,yhat) + l1_alpha*torch.norm(weights,1)
    return loss
def Numparams(weights):
    l1 = torch.sum(torch.abs(weights))
    nparms = torch.sum(torch.abs(weights)>0)

    return nparms.cpu().detach(), l1.cpu().detach()

#Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(net.parameters(),lr = 0.0003)

print(train_X.shape,train_Y.shape,val_X.shape,val_Y.shape)
data = [[],[]]
best_val_epoch = 0
best_val = float('Inf')
for e in range(epochs):

    # Train data

    train_loss = 0
    batch_count = 0
    for i in range(len(train_X)//batch_size+1):
        if i*batch_size >= len(train_X):continue
        x = torch.tensor(train_X[i*batch_size:(i+1)*batch_size]).to(computing_device)
        if multiclass:
            y = torch.tensor(train_Y[i*batch_size:(i+1)*batch_size]).to(computing_device)
        else:
            y = torch.tensor(train_Y[i*batch_size:(i+1)*batch_size]).view(-1,1).to(computing_device)
        optimizer.zero_grad()

        outputs = net(x.float())
        #print(x.shape,outputs.shape,y.shape)
        #train_pred.append(outputs.item())
        #print(outputs.shape,y.float().shape)
        loss = torch.sqrt(criterion(outputs,y.float(),net.main.weight))
        #print(loss)
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
    data[0].append(train_loss/batch_count)
    # Validate data
    val_loss = 0
    val_pred = []
    batch_count = 0
    for i in range(len(val_X)//batch_size+1):

        with torch.no_grad():
            x = torch.tensor(val_X[i*batch_size:(i+1)*batch_size]).to(computing_device)
            if multiclass:
                y = torch.tensor(val_Y[i*batch_size:(i+1)*batch_size]).to(computing_device)
            else:
                y = torch.tensor(val_Y[i*batch_size:(i+1)*batch_size]).view(-1,1).to(computing_device)
            outputs = net(x.float())
            loss = torch.sqrt(criterion(outputs,y.float(),net.main.weight))
            val_loss += loss.item()
            batch_count += 1

    if not multiclass:
        val_result = evaluateCNN(net,val_X,val_Y,computing_device)
        print('\rEpoch {}, Val Loss: {}, Val R2 Score:{}'.format(e,val_loss/batch_count, val_result[1]))
        data[1].append(val_loss/batch_count)
        if pretrained:
            np.save('results/pytorchResultsPretrainedFilters{}Size{}.npy'.format(num_features_selected,num_filters), np.array(data))
        else:
            np.save('results/pytorchResultsFilters{}Size{}.npy'.format(num_features_selected,filter_size), np.array(data))

        print(val_loss/batch_count,best_val,e,best_val_epoch)
        if val_loss/batch_count < best_val:
            best_val = val_loss/batch_count
            best_val_epoch = e
            if pretrained:
                torch.save(net.state_dict(),'model_files/PretrainedFilters{}Size{}.pt'.format(num_features_selected,num_filters))
            else:
                torch.save(net.state_dict(),'model_files/Filters{}Size{}.npy'.format(num_features_selected,filter_size))
        elif e  - best_val_epoch > 10:
            print("Stop, Best Validation:{:.4f}, Best Validation Epoch:{}".format(best_val,best_val_epoch))
            break

        #val_result = evaluateCNN(net,val_X,val_Y,computing_device)
        #print('\rEpoch {}, Val Loss: {}, Val R2 Score:{}'.format(e,val_loss/batch_count, val_result[1]))
    else:
        evaluateMultiClassCNN(net,val_X,val_Y,computing_device)
