#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:31:20 2024

@author: fuzhu
"""

import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial

##Torch imports
import torch.nn.functional as F
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
import pickle, json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('training_log.txt')])
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data):
        features_tensor = data['features']
        targets_tensor = data['labels']
        energies_tensor = data['energies']
        self.features_tensor = torch.tensor(np.array(features_tensor), dtype=torch.float32)
        self.targets_tensor = torch.tensor(np.array(targets_tensor), dtype=torch.float32)
        self.energies_tensor = torch.tensor(np.array(energies_tensor), dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.features_tensor[index], self.targets_tensor[index], self.energies_tensor[index]
    
    def __len__(self):
        return self.features_tensor.size(0)


class CustomDataset_predict(Dataset):
    def __init__(self, data):
        features_tensor = data['features']
        self.features_tensor = torch.tensor(np.array(features_tensor), dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.features_tensor[index]
    
    def __len__(self):
        return self.features_tensor.size(0)   

def high_order_moment_based_loss(predictions, targets):
    # High-Order Moment-Based Loss: loss.item()
    # Calculate moments for predictions
    pred_mean = torch.mean(predictions)
    pred_var = torch.var(predictions, unbiased=False)
    pred_skew = torch.mean(((predictions - pred_mean) / torch.sqrt(pred_var))**3)
    pred_kurt = torch.mean(((predictions - pred_mean) / torch.sqrt(pred_var))**4) - 3

    # Calculate moments for targets
    target_mean = torch.mean(targets)
    target_var = torch.var(targets, unbiased=False)
    target_skew = torch.mean(((targets - target_mean) / torch.sqrt(target_var))**3)
    target_kurt = torch.mean(((targets - target_mean) / torch.sqrt(target_var))**4) - 3

    # Calculate moment losses
    mean_loss = (pred_mean - target_mean) ** 2
    var_loss = (pred_var - target_var) ** 2
    skew_loss = (pred_skew - target_skew) ** 2
    kurt_loss = (pred_kurt - target_kurt) ** 2

    # Combine losses
    total_loss = mean_loss + var_loss + skew_loss + kurt_loss

    return total_loss

##get dos features
def get_dos_band_features(energies, dos):
    #dos = torch.tensor(dos, dtype=torch.float32)
    #energies = torch.tensor(energies, dtype=torch.float32)
    dos=torch.abs(dos)
    center=torch.sum(energies*dos, axis=1)/torch.sum(dos, axis=1)
    #print (energies.size(), center[:, None].size())
    E_offset = energies - center[:, None]
    #print (energies.size(), E_offset.size(), dos.size())
    width = torch.diagonal(torch.mm((E_offset**2), dos.T))/torch.sum(dos, axis=1)
    skew = torch.diagonal(torch.mm((E_offset**3), dos.T))/torch.sum(dos, axis=1)/width**(1.5)
    kurtosis = torch.diagonal(torch.mm((E_offset**4), dos.T))/torch.sum(dos, axis=1)/width**(2)
    
    #find zero index (fermi leve)
    #zero_index = torch.abs(energies-0).argmin().long()
    #genetal_band_center=torch.sum(energies[:,:zero_index] * dos[:,:zero_index], axis=1)/torch.sum(dos[:,:zero_index], axis=1)
    return torch.stack((center, width, skew, kurtosis), axis=1)

# Example usage
#predictions = torch.tensor([2.3, 2.5, 2.7, 2.9], dtype=torch.float32)
#targets = torch.tensor([2.0, 2.0, 3.0, 3.0], dtype=torch.float32)
#
#loss = high_order_moment_based_loss(predictions, targets)
#print("High-Order Moment-Based Loss:", loss.item())

def calculate_moments(tensor):
    mean = torch.mean(tensor)
    var = torch.var(tensor, unbiased=False) + 1e-6 #add small value to avoid division by zero
    skew = torch.mean(((tensor - mean) / torch.sqrt(var))**3)
    kurt = torch.mean(((tensor - mean) / torch.sqrt(var))**4) - 3
    return torch.tensor([mean, var, skew, kurt])

def calculate_loss(dos_out, dos_target, energy, loss_method, loss_feature):
    dos_loss = getattr(F, loss_method)(dos_out, dos_target)
    dos_out_cumsum = torch.cumsum(dos_out, axis=1)
    dos_cumsum = torch.cumsum(dos_target, axis=1)
    dos_cumsum_loss = getattr(F, loss_method)(dos_out_cumsum, dos_cumsum)
    pred_moments = calculate_moments(dos_out)
    target_moments = calculate_moments(dos_target)
    dos_moment_loss = getattr(F, loss_method)(pred_moments, target_moments)
    band_center_out = get_dos_band_features(energy, dos_out)
    band_center = get_dos_band_features(energy, dos_target)
    loss_band_center = getattr(F, loss_method)(band_center, band_center_out)
    if loss_feature:
        loss_sum = dos_loss + 0.05 * dos_cumsum_loss + 0.2 * loss_band_center + 0.2 * dos_moment_loss
    else:
        loss_sum = dos_loss
    return loss_sum

def calculate_rmse(dos_out_predicted, dos_target):
    """
    Calculate the Root Mean Squared Error (RMSE) between predicted and target DOS values.

    Args:
        dos_out_predicted (torch.Tensor): Predicted DOS values.
        dos_target (torch.Tensor): Target DOS values.

    Returns:
        torch.Tensor: The RMSE value.
    """
    mse = torch.mean((dos_out_predicted - dos_target) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def train(data_loader, model, optimizer, loss_method, pinn_loss, rank):
    model.train()
    loss_total = 0
    rmse_total = 0
    count = 0
    for feature, dos_target, energies in data_loader:
        #print (dos_target)
        feature = feature.to(rank)  ### rank ['cpu', 'cuda']
        dos_target = dos_target.to(rank)  ### rank ['cpu', 'cuda']
        energies = energies.to(rank)
        optimizer.zero_grad()
        
        assert not torch.isinf(feature).any(), "Input contains infinite values"
        assert not torch.isnan(feature).any(), "Input contains nan values"
        
        descriptor, dos_out = model(dos_target)
        
        #===================define loss===============================================================
        #--------------------------------------loss_dos-----------------------------------------------
        dos_loss = getattr(F, loss_method)(dos_out, dos_target)
        feature_loss = getattr(F, loss_method)(descriptor, feature)
        
        #--------------------------------------loss_cumsum-----------------------------------------------
        dos_out_cumsum = torch.cumsum(dos_out, axis=1)
        dos_cumsum = torch.cumsum(dos_target, axis=1)
        dos_cumsum_loss = getattr(F, loss_method)(dos_out_cumsum, dos_cumsum)

        #--------------------------------------loss_moments-----------------------------------------------
        # Calculate moments for predictions: dos_out
        pred_mean = torch.mean(dos_out)
        pred_var = torch.var(dos_out, unbiased=False)
        pred_skew = torch.mean(((dos_out - pred_mean) / torch.sqrt(pred_var))**3)
        pred_kurt = torch.mean(((dos_out - pred_mean) / torch.sqrt(pred_var))**4) - 3
    
        # Calculate moments for targets: data.scaled_dos
        target_mean = torch.mean(dos_target)
        target_var = torch.var(dos_target, unbiased=False)
        target_skew = torch.mean(((dos_target - target_mean) / torch.sqrt(target_var))**3)
        target_kurt = torch.mean(((dos_target - target_mean) / torch.sqrt(target_var))**4) - 3
        
        pred_moments = torch.tensor([pred_mean, pred_var, pred_skew, pred_kurt])
        dft_moments = torch.tensor([target_mean, target_var, target_skew, target_kurt])
        dos_moment_loss = getattr(F, loss_method)(pred_moments, dft_moments)
        
        #--------------------------------------loss_d-band-center-----------------------------------------------
        band_center_out = get_dos_band_features(energies, dos_out)
        band_center = get_dos_band_features(energies, dos_target)
        band_center = band_center.to(rank)
        loss_band_center = getattr(F, loss_method)(band_center, band_center_out.to(band_center.device))
        #===================================================================================================================================================
    
        if np.isnan(band_center.detach()).any():
            print (band_center)
            raise ValueError("Input arrays contain NaN values of d_band_dft.")
         
            
        if pinn_loss:
            loss_sum = dos_loss + feature_loss + 0.2 * dos_cumsum_loss + 0.2 * loss_band_center + 0.2 * dos_moment_loss
        else:
            loss_sum = dos_loss + feature_loss # + 0.05 * dos_cumsum_loss + 0.2 * loss_band_center + 0.2 * dos_moment_loss
        
        loss_total += loss_sum.detach() * dos_out.size(0)
        loss_sum.backward()
        

        rmse = calculate_rmse(dos_out, dos_target)
        rmse_total += rmse.detach() * dos_out.size(0)
        
        optimizer.step()
        count += dos_out.size(0)
    
    count = torch.tensor(count, dtype=torch.float32)
    #print (loss_total, count)
    loss_total = loss_total / count
    return loss_total, rmse_total
        

def validate(validate_loader, model, loss_method, pinn_loss, rank):
    model.eval()
    loss_total = 0
    rmse_total = 0
    count = 0
    with torch.no_grad():
        for feature, dos_target, energies in validate_loader:
            feature = feature.to(rank)  ### rank ['cpu', 'cuda']
            dos_target = dos_target.to(rank)  ### rank ['cpu', 'cuda']
            energies = energies.to(rank)
            
            descriptor, dos_out = model(dos_target)
            
            #===================define loss===============================================================
            #--------------------------------------loss_dos-----------------------------------------------
            dos_loss = getattr(F, loss_method)(dos_out, dos_target)
            feature_loss = getattr(F, loss_method)(descriptor, feature)
            
            #--------------------------------------loss_cumsum-----------------------------------------------
            dos_out_cumsum = torch.cumsum(dos_out, axis=1)
            dos_cumsum = torch.cumsum(dos_target, axis=1)
            dos_cumsum_loss = getattr(F, loss_method)(dos_out_cumsum, dos_cumsum)
    
            #--------------------------------------loss_moments-----------------------------------------------
            # Calculate moments for predictions: dos_out
            pred_mean = torch.mean(dos_out)
            pred_var = torch.var(dos_out, unbiased=False)
            pred_skew = torch.mean(((dos_out - pred_mean) / torch.sqrt(pred_var))**3)
            pred_kurt = torch.mean(((dos_out - pred_mean) / torch.sqrt(pred_var))**4) - 3
        
            # Calculate moments for targets: data.scaled_dos
            target_mean = torch.mean(dos_target)
            target_var = torch.var(dos_target, unbiased=False)
            target_skew = torch.mean(((dos_target - target_mean) / torch.sqrt(target_var))**3)
            target_kurt = torch.mean(((dos_target - target_mean) / torch.sqrt(target_var))**4) - 3
            
            pred_moments = torch.tensor([pred_mean, pred_var, pred_skew, pred_kurt])
            dft_moments = torch.tensor([target_mean, target_var, target_skew, target_kurt])
            dos_moment_loss = getattr(F, loss_method)(pred_moments, dft_moments)
            
            #--------------------------------------loss_d-band-center-----------------------------------------------
            #E = torch.linspace(-10, 10, 1000).to(scaled_dos_out)
            #band_center = get_dos_band_features(E, data.scaled_dos)
            #print (energies.size(), scaled_dos_out.size())
            band_center_out = get_dos_band_features(energies, dos_out)
            band_center = get_dos_band_features(energies, dos_target)
            band_center = band_center.to(rank)
            loss_band_center = getattr(F, loss_method)(band_center, band_center_out.to(band_center.device))
            #===================================================================================================================================================
            
            if pinn_loss:
                loss_sum = dos_loss + feature_loss + 0.2 * dos_cumsum_loss + 0.2 * loss_band_center + 0.2 * dos_moment_loss
            else:
                loss_sum = dos_loss + feature_loss # + 0.05 * dos_cumsum_loss + 0.2 * loss_band_center + 0.2 * dos_moment_loss
            
            loss_total += loss_sum.detach() * dos_out.size(0)
    
            rmse = calculate_rmse(dos_out, dos_target)
            rmse_total += rmse.detach() * dos_out.size(0)
            
            count = count + dos_out.size(0)
        
        count = torch.tensor(count, dtype=torch.float32)
        loss_total = loss_total / count

    return loss_total, rmse_total
    

def train_model(train_loader, validate_loader, model, optimizer, loss_method='mse_loss', pinn_loss=False, epochs=2000, verbosity=100, filename='checkpoint.pth.tar'):
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(1, epochs+1):
        #lr = scheduler.optimizer.param_groups[0]['lr']
        # Train model
        train_loss_total, train_rmse_total = train(train_loader, model, optimizer, loss_method, pinn_loss, rank)
        test_loss_total, test_rmse_total = validate(validate_loader, model, loss_method, pinn_loss, rank)
        
        #best_model = copy.deepcopy(model.module)
        state = {"state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "full_model": model}
        torch.save(state, filename)
        
        #loss_diff = torch.abs(train_loss_total - test_loss_total)
        if test_loss_total < best_loss:
            best_loss = test_loss_total
            best_epoch = epoch
            torch.save(state, 'best.pth.tar')

        torch.save(state, filename)  # Save checkpoint at each epoch

        if epoch % verbosity == 0:
            logger.info(f'Epoch [{epoch+1:04d}/{epochs:04d}], TrainLoss: {train_loss_total:.4f}, ValLoss: {test_loss_total:.4f}, TrainRMSE: {train_rmse_total:.4f}, ValRMSE: {test_rmse_total:.4f}')
    
    print(f"Best validation loss: {best_loss:.4f} at epoch {best_epoch}")
    return model

def predict(dataset, model_path, batch_size, num_workers=0):
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    load_path = os.path.join(model_path, 'best.pth.tar')    
    saved_model = torch.load(load_path, map_location=torch.device(rank), weights_only=False)
    model = saved_model['full_model']
    model = model.to(rank)
    
    model.eval()
    pdos_predictions = []
    d_band_predictions = []
    pdos_dft = []
    d_band_dft = []
    energies = []
    with torch.no_grad():
        for features, dos_target, energy in dataloader:
            #features = features.to(rank)
            assert not torch.isinf(features).any(), "Input contains infinite values"
            assert not torch.isnan(features).any(), "Input contains nan values"
            dos_out = model.get_Property(features)
            pdos_predictions.extend(dos_out.cpu().numpy())
            #---------------------------band-center---------
            band_center_out = get_dos_band_features(energy, dos_out)
            d_band_predictions.extend(band_center_out.cpu().numpy())
            
            if np.isnan(d_band_predictions).any():
                raise ValueError("Input arrays contain NaN values of d_band_predictions.")
                    
            #-----------------------------------------------------------------
            pdos_dft.extend(dos_target.cpu().numpy())
            band_center_dft = get_dos_band_features(energy, dos_target)
            d_band_dft.extend(band_center_dft.cpu().numpy())
            energies.extend(energy.cpu().numpy())
    
    with open('Predicted_pDOS.pkl', 'wb') as f:
        pickle.dump(pdos_predictions, f)
        
    with open('Prediced_band.pkl', 'wb') as f:
        pickle.dump(d_band_predictions, f)
    
    return pdos_predictions, d_band_predictions, pdos_dft, d_band_dft, energies

def predict_new(dataset, model_path, batch_size, num_workers=0):
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    load_path = os.path.join(model_path, 'best.pth.tar')    
    saved_model = torch.load(load_path, map_location=torch.device(rank), weights_only=False)
    model = saved_model['full_model']
    model = model.to(rank)
    
    model.eval()
    pdos_predictions = []

    with torch.no_grad():
        for features  in dataloader:
            #features = features.to(rank)
            assert not torch.isinf(features).any(), "Input contains infinite values"
            assert not torch.isnan(features).any(), "Input contains nan values"
            dos_out = model.get_Property(features)
            pdos_predictions.extend(dos_out.cpu().numpy())
    
    with open('Predicted_pDOS.pkl', 'wb') as f:
        pickle.dump(pdos_predictions, f)
    
    return pdos_predictions
