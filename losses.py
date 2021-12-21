#!/usr/bin/env python3
# encoding: utf-8
import torch
from torch.nn import functional as F
import numpy as np
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image 
import cv2
import torch.nn as nn
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency = 10, consistency_rampup = 20.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
  
def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def dice_loss(input, target):
    """soft dice loss"""
    eps = 1e-7
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)


def vae_loss(recon_x, x, mu, logvar):
    loss_dict = {}
    loss_dict['KLD'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss_dict['recon_loss'] = F.mse_loss(recon_x, x, reduction='mean')

    return loss_dict

def unet_vae_loss(cfg, batch_pred, batch_x, batch_y, vout, mu, logvar):
    loss_dict = {}
    loss_dict['ed_loss'] = dice_loss(batch_pred[:, 0], batch_y[:, 0])  # ed
    loss_dict['net_loss'] = dice_loss(batch_pred[:, 1], batch_y[:, 1])  # net
    loss_dict['et_loss'] = dice_loss(batch_pred[:, 2], batch_y[:, 2])  # et enhance tumor
    loss_dict.update(vae_loss(vout, batch_x, mu, logvar))
    weight = cfg.MODEL.LOSS_WEIGHT
    loss_dict['loss'] = loss_dict['ed_loss'] + loss_dict['net_loss'] + loss_dict['et_loss'] 

    return loss_dict

#### Consistency loss ###
  
def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    mse_mean_loss = torch.mean(mse_loss)
    return mse_mean_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    #print("kl_div shape:", kl_div.shape)
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])

    return kl_div
    
#### Entropy loss ###
def prob_2_entropy(v):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
     #([1, 4, 160, 192, 128]) CHWD
    assert v.dim() == 5
    n, c, h, w, d= v.size()
    prob = v
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

#### loss_Co-training ###
"""
def vae_UDA_loss(recon_x, x, mu, logvar, recon_x_mono, x_mono, mu_mono, logvar_mono):
    loss_dict = {}
    loss_dict['KLD'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss_dict['recon_loss'] = F.mse_loss(recon_x, x, reduction='mean')
    loss_dict['KLD_mono'] = -0.5 * torch.sum(1 + logvar_mono - mu_mono.pow(2) - logvar_mono.exp())
    loss_dict['recon_mono_loss'] = F.mse_loss(recon_x_mono, x_mono, reduction='mean')

    return loss_dict
"""
def unet_Co_loss(cfg, batch_pred, batch_x, batch_y, vout, mu, logvar, batch_pred_mono, batch_x_mono, vout_mono, mu_mono, logvar_mono, epoch):
    loss_dict = {}
    loss_dict['ed_dc_loss'] = dice_loss(batch_pred[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['net_dc_loss'] = dice_loss(batch_pred[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_dc_loss'] = dice_loss(batch_pred[:, 2], batch_y[:, 2])  # enhance tumor
    
    loss_dict['ed_mono_dc_loss'] = dice_loss(batch_pred_mono[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['net_mono_dc_loss'] = dice_loss(batch_pred_mono[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_mono_dc_loss'] = dice_loss(batch_pred_mono[:, 2], batch_y[:, 2])  # enhance tumor
    
    #loss_dict.update(vae_UDA_loss(vout, batch_x, mu, logvar, vout_mono, batch_x_mono, mu_mono, logvar_mono))
    
    
    loss_dict['loss_dc'] = loss_dict['ed_dc_loss'] + loss_dict['net_dc_loss'] + loss_dict['et_dc_loss']
    loss_dict['loss_mono_dc'] = loss_dict['ed_mono_dc_loss'] + loss_dict['net_mono_dc_loss'] + loss_dict['et_mono_dc_loss']
    
    if cfg.consistency_type == 'mse':
        loss_dict['ed_mse_loss'] = F.mse_loss(batch_pred_mono[:, 0], batch_pred[:, 0], reduction='mean') 
        loss_dict['net_mse_loss'] = F.mse_loss(batch_pred[:, 1], batch_pred_mono[:, 1], reduction='mean') 
        loss_dict['et_mse_loss'] = F.mse_loss(batch_pred[:, 2], batch_pred_mono[:, 2], reduction='mean') 
        loss_dict['consistency_loss'] = loss_dict['ed_mse_loss'] + loss_dict['net_mse_loss'] + loss_dict['et_mse_loss']
    elif cfg.consistency_type == 'kl':
        batch_pred_mono_softmax = F.log_softmax(batch_pred_mono, dim=1)
        batch_pred_softmax = F.softmax(batch_pred, dim=1)
        loss_dict['consistency_loss'] = F.kl_div(batch_pred_mono_softmax, batch_pred_softmax, reduction='mean')
    else:
        assert False, cfg.consistency_type
  
    seg_volume = torch.zeros([batch_y.shape[0], 1, batch_y.shape[2], batch_y.shape[3], batch_y.shape[4]], dtype = torch.long)
    seg_volume = (batch_y[:, 0] + batch_y[:, 1] + batch_y[:, 2]).long()
    
    weight_mono = cfg.MODEL.LOSS_WEIGHT_MONO
    weight_main = 1 - cfg.MODEL.LOSS_WEIGHT_MONO
    
    weight_consistency = get_current_consistency_weight(epoch)
    loss_dict['loss_Co'] = weight_main * loss_dict['loss_dc'] + weight_mono * loss_dict['loss_mono_dc'] + \
                            weight_consistency * loss_dict['consistency_loss']
    
    return loss_dict

def get_losses(cfg):
    losses = {}
    losses['vae'] = vae_loss
    losses['dice'] = dice_loss
    losses['dice_vae'] = unet_vae_loss
    losses['dice_ACN'] = unet_Co_loss
    return losses
