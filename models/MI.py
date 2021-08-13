import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#### Modality-mutual information knowledge transfer  #####
def Cal_MIloss(input, target, pred_mean, log_scale, init_pred_var=5.0, eps=1e-5):
        # pool for dimentsion match
        
        s_H, t_H = input.shape[2], target.shape[2]
        s_W, t_W = input.shape[3], target.shape[3]
        s_D, t_D = input.shape[4], target.shape[4]
        assert(s_H == t_H)
        
        pred_var = torch.log(1.0+torch.exp(log_scale))+ eps
        pred_var = pred_var.view(1, -1, 1, 1, 1)
        
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss