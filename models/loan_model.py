import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet
import torch.nn.functional as F
import numpy as np
import pdb


class LoanNet(SimpleNet):
    def __init__(self, in_dim=90,  out_dim=9, name=None, created_time=None):
        super(LoanNet, self).__init__(f'{name}_Simple', created_time)
      
        self.fc_layer = torch.nn.Sequential(
            nn.Linear(90, 9),
        )


    def forward(self, x):

        x = self.fc_layer(x)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return x