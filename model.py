import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim




class Net(nn.Module):

    def __init__(self, input_dim,output_dim):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, output_dim)# 6*6 from image dimension
        #self.fc2 = nn.Linear(100, 16)
        #self.fc3 = nn.Linear(16, output_dim)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.tanh(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
