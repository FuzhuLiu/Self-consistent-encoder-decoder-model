#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:28:47 2024

@author: fuzhu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset


class PFP_Network(nn.Module):
    def __init__(self, P_channels, hidden_layers_nodes_1, hidden_layers_nodes_2, F_channels, dropout=False, dropout_rate=0.2):
        super(PFP_Network, self).__init__()
        # hidden_layers_nodes=[10,10]
        FC_1 = []
        input_size = P_channels
        for hidden_size in hidden_layers_nodes_1:
            fc = nn.Linear(input_size, hidden_size)
            FC_1.append(fc)
            if dropout:
                FC_1.append(nn.Dropout(p=dropout_rate))
            FC_1.append(nn.PReLU())
            input_size = hidden_size
            
        self.FC_1 = nn.ModuleList(FC_1)
        self.feature = nn.Sequential(nn.Linear(hidden_layers_nodes_1[-1], F_channels))

        FC_2 = []
        input_size = F_channels
        for hidden_size in hidden_layers_nodes_2:
            fc = nn.Linear(input_size, hidden_size)
            FC_2.append(fc)
            if dropout:
                FC_2.append(nn.Dropout(p=dropout_rate))
            FC_2.append(nn.PReLU())
            input_size = hidden_size
            
        self.FC_2 = nn.ModuleList(FC_2)
        self.property = nn.Sequential(nn.Linear(hidden_layers_nodes_2[-1], P_channels))
        
    def forward(self, x):
        feature = self.get_Feature(x)
        dos_out = self.get_Property(feature)
        return feature, dos_out
    
    def get_Feature(self, x):
        for hidden_layer in self.FC_1:
            x = hidden_layer(x)
        Feature = self.feature(x)
        return Feature
    
    def get_Property(self, x):
        for hidden_layer in self.FC_2:
            x = hidden_layer(x)
        Property = self.property(x)
        return Property
        


