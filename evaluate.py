import torch
import torch.nn.functional as func
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

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

            pred.append(output.item())

    #return math.sqrt(mean_squared_error(pred,levels)),r2_score(levels,pred)


