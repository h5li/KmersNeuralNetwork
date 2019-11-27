import torch
import numpy as np
import torch.nn.functional as func
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import sys

def evaluate(net,data,levels,computing_device):

    pred = []
    for x in data:
        with torch.no_grad():
            x = torch.tensor(x).to(computing_device)
            output = net(x.float())

            pred.append(output.item())

    return math.sqrt(mean_squared_error(pred,levels)),r2_score(levels,pred)

def evaluateMultiClass(net,data,levels,computing_device):

    pred = []
    for x in data:
        with torch.no_grad():
            x = torch.tensor(x).to(computing_device)
            output = net(x.float())
            pred.append(output.cpu().numpy())
    pred = np.array(pred)
    score_sum = 0
    for i in range(levels.shape[1]):
        sys.stdout.write("T:{} S:{:.4f} ".format(i,r2_score(levels[:,i],pred[:,i])))
        score_sum += r2_score(levels[:,i],pred[:,i]) 
    sys.stdout.write(str(max(0,score_sum)))
    sys.stdout.write("\n")
    #return math.sqrt(mean_squared_error(pred,levels)),r2_score(levels,pred)



def evaluateMultiClassCNN(net,data,levels,computing_device):

    pred = []
    for x in data:
        with torch.no_grad():
            x = torch.tensor(x).view(1,4,600).to(computing_device)
            output = net(x.float())
            pred.append(output.cpu().numpy())
    pred = np.array(pred)
    score_sum = 0
    for i in range(levels.shape[1]):
        sys.stdout.write("T:{} S:{:.4f} ".format(i,r2_score(levels[:,i],pred[:,0,i])))
        score_sum += r2_score(levels[:,i],pred[:,0,i]) 
    sys.stdout.write(str(max(0,score_sum)))
    sys.stdout.write("\n")
    #return math.sqrt(mean_squared_error(pred,levels)),r2_score(levels,pred)


