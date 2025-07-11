#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:43:18 2023

@author: fuzhu
"""

import torch
import time
from Training_model import CustomDataset, train_model, predict
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PFP_Network import PFP_Network
from sklearn import metrics

#==================================USAGE======================================================            
print ("Starting to train a model ...")
start_time = time.time()
#-----------------------------Model parameters----------------------------------------------
P_channels = 2000
hidden_layers_nodes_1 = [1400, 1296]
hidden_layers_nodes_2 = [1296, 1400]
F_channels = 1296
dropout = False
dropout_rate = 0.01
rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PFP_Network(P_channels, hidden_layers_nodes_1, hidden_layers_nodes_2, F_channels, dropout, dropout_rate)
model.to(rank)
print (model)
#========================================================================================
#----------------------------------load dataset------------------------------------------
Data_xy = torch.load('Data_3E.pth', weights_only=False)
dataset = CustomDataset(Data_xy)
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size], torch.Generator().manual_seed(42))

batch_size = 90
dataset_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataset_validate = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print ('*' * 100)

#---------------------------------traning paramters---------------------------------
epochs = 1000
loss_method = "l1_loss"
pinn_loss = False
verbosity = 10
lr = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
X = train_model(dataset_train, dataset_validate, model, optimizer, loss_method=loss_method, loss_feature=loss_feature, epochs=epochs, verbosity=verbosity, filename='checkpoint.pth.tar')
end_time = time.time()
total_time = end_time - start_time
print ('total_time: {:.4f}, Done!'.format(total_time))


#-#-----------------------------validate a model----------------------------------------
Data_xy = torch.load('Data_2E.pth', weights_only=False)
dataset = CustomDataset(Data_xy)
model_path = os.getcwd()
pdos_predictions, d_band_predictions, pdos_dft, d_band_dft, energies = predict(dataset, model_path, batch_size, num_workers=0)


R2_train = metrics.r2_score(pdos_predictions, pdos_dft)
mse_train = metrics.mean_squared_error(pdos_predictions, pdos_dft)
print (R2_train, mse_train)
R2_test = metrics.r2_score(d_band_predictions, d_band_dft)
mse_test = metrics.mean_squared_error(d_band_predictions, d_band_dft)
print (R2_test, mse_test)

plt.scatter(pdos_predictions, pdos_dft, c='r', label='train')
plt.scatter(d_band_predictions, d_band_dft, c='b', label='test')
plt.xlabel('DOS$_{ML}$ (a.u.)')
plt.ylabel('DOS$_{DFT}$ (a.u.)')
plt.legend(loc='best')
plt.savefig('parity.png')
#plt.show()

index = 6
fig, ax = plt.subplots(figsize=(5,4))
ax.tick_params(axis='both', direction='in', labelsize=14)
ax.plot(range(len(pdos_dft[index])), pdos_dft[index], '-', color='g', label='DFT')
ax.plot(range(len(pdos_predictions[index])), pdos_predictions[index], '-', color='r', label='ML')

#ax.plot(N_U_ratio_p, p_band_centers, '-o', color='k', label='N p band')
#ax.set_xlim(-6, 4)
#ax.set_ylim(-4.1, -2.2)
ax.legend(loc='best')
ax.set_xlabel('Energy (eV)', fontsize=14)
ax.set_ylabel('pDOS (a.u.)', fontsize=14)
fig.tight_layout()
plt.savefig('example_dos.png', facecolor='w', dpi=600)
#plt.show()
