import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EmnistNet(SimpleNet):#model for mnist
    def __init__(self, name=None, created_time=None,num_of_classes = 47):
        super(EmnistNet, self).__init__(f'{name}_Simple', created_time)
        self.fc_layer = torch.nn.Sequential(#1 * 28 * 28
            Flatten(),#784
            nn.Linear(784, num_of_classes),
        )
    
    def forward(self, x):
        out = self.fc_layer(x)
        return out
 