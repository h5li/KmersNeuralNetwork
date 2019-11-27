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
        self.fc1 = nn.Linear(input_dim, 100)# 6*6 from image dimension
        #self.fc2 = nn.Linear(100, 16)
        self.fc3 = nn.Linear(100, output_dim)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

num_filters = [21,20,15,7]
filter_size = [3,3,3,5]
stride = [2,2,2,2]
padding = [1,1,1,1]

class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(4, num_filters[0], 
                        filter_size[0], 
                        stride=stride[0], 
                        padding=padding[0], 
                        bias=False), 

            nn.ReLU(inplace=True),

            nn.Conv1d(num_filters[0], 
                      num_filters[1], 
                      filter_size[1],
                      stride=stride[1], 
                      padding=padding[1], 
                      bias=False),

            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),

            nn.Conv1d(num_filters[1],  
                      num_filters[2], 
                      filter_size[2], 
                      stride=stride[2], 
                      padding=padding[2], 
                      bias=False),
                    
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),

            nn.Conv1d(num_filters[2], 
                      num_filters[3], 
                      filter_size[3], 
                      stride=stride[3], 
                      padding=padding[3], 
                      bias=False),

            nn.BatchNorm2d(num_filters[3]),

            nn.ReLU(inplace=True),
        )
        def getCNNOutputDim(InputSize, Padding, KernalSize, Stride):
            return ((InputSize + 2 * Padding - KernalSize)//Stride + 1)
        
        def finalOutputDim():
            inputSize = 224

            for i in range(len(num_filters)):
                inputSize = getCNNOutputDim(inputSize,padding[i],filter_size[i],stride[i])
            return inputSize * num_filters[-1]

        # CNN Output Dim = (Size + 2 * padding - kernal_size)/Stride + 1
        self.fc = nn.Sequential(
            nn.Linear(finalOutputDim(),600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 16)
        )

    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, self.num_flat_features(x))
        return self.fc(x)