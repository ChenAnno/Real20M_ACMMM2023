from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class Product(nn.Module):
    def __init__(self, in_features, out_features):
        super(Product, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, input, label):
        output = self.fc(input)
        return output

