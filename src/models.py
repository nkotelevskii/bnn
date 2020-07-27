import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_classification(nn.Module):
    def __init__(self, args):
        super(Net_classification, self).__init__()
        last_features = args.last_features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.linear1 = nn.Linear(in_features=1024, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=last_features)

    def forward(self, x):
        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(h1))
        h3 = torch.relu(self.conv3(h2))
        h3_flat = h3.view(h3.shape[0], -1)
        h4 = torch.relu(self.linear1(h3_flat))
        h5 = torch.relu(self.linear2(h4))
        return h5
    
    
    
class Net_regression(nn.Module):
    def __init__(self, args):
        super(Net_regression, self).__init__()
        in_features = args.in_features
        last_features = args.last_features
        self.linear1 = nn.Linear(in_features=in_features, out_features=10*in_features)
        self.linear2 = nn.Linear(in_features=10*in_features, out_features=last_features)

    def forward(self, x):
        h3_flat = x
        h4 = torch.relu(self.linear1(h3_flat))
        h5 = torch.relu(self.linear2(h4))
        return h5
    
class Dropout_layer(nn.Module):
    def __init__(self,):
        super(Dropout_layer, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        return self.dropout(x)