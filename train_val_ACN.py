#!/usr/bin/env python3
# encoding: utf-8
from config import _C as cfg
from data import make_data_loaders
from models import build_model
from models.discriminator import get_fc_discriminator, get_df_discriminator
from models.MI import Cal_MIloss
from utils import init_env, mkdir
from solver import make_optimizer, make_optimizer_double
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.logger import setup_logger
from utils.metric_logger import MetricLogger
import logging
import time
from losses import get_losses, bce_loss, prob_2_entropy
from metrics import get_metrics
import shutil
import nibabel as nib
import numpy as np
from PIL import Image 
from torchvision.utils import make_grid
from utils import viz_segmask

def load_old_model(model, model_mono, d_main, optimizer, saved_model_path):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint["model"])
    model_mono.load_state_dict(checkpoint["model_mono"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    d_main.load_state_dict(checkpoint["d_main"])
    return model, model_mono, d_main, optimizer 
        
def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def save_sample(batch_pred, batch_pred_mono, batch_x, batch_y, epoch, batch_id):
    def get_mask(seg_volume):
        seg_volume = seg_volume.cpu().numpy()
        seg_volume = np.squeeze(seg_volume)
        wt_pred = seg_volume[0]
        tc_pred = seg_volume[1]
        et_pred = seg_volume[2]
        mask = np.zeros_like(wt_pred)
        mask[wt_pred > 0.5] = 2
        mask[tc_pred > 0.5] = 1
        mask[et_pred > 0.5] = 4
        mask = mask.astype("uint8")
        mask_nii = nib.Nifti1Image(mask, np.eye(4))
        return mask_nii

    volume = batch_x[:, 0].cpu() # only save one modality
    volume = (volume.numpy()[0] * 255).astype('uint8')
    volume_nii = nib.Nifti1Image(volume, np.eye(4))
    log_dir = os.path.join(cfg.LOG_DIR, cfg.TASK_NAME, 'epoch'+str(epoch))
    mkdir(log_dir)
    nib.save(volume_nii, os.path.join(log_dir, 'batch'+str(batch_id)+'_volume.nii.gz'))
    pred_nii = get_mask(batch_pred)
    pred_nii_mono = get_mask(batch_pred_mono)
    gt_nii = get_mask(batch_y)
    nib.save(pred_nii, os.path.join(log_dir, 'batch' + str(batch_id) + '_pred.nii.gz'))
    nib.save(pred_nii_mono, os.path.join(log_dir, 'batch' + str(batch_id) + '_pred_mono.nii.gz'))
    nib.save(gt_nii, os.path.join(log_dir, 'batch' + str(batch_id) + '_gt.nii.gz'))

def train_val(model, model_mono, d_main, d_feature, loaders, optimizer, scheduler, losses, metrics=None, epoch_init=0):
    n_epochs = cfg.SOLVER.NUM_EPOCHS
    iter_num = 0
    end = time.time()
    best_dice = 0.0

    for epoch in range(epoch_init, n_epochs):
        scheduler.step()
        for phase in ['train', 'eval']:
            meters = MetricLogger(delimiter=" ")
            loader = loaders[phase]
            getattr(model, phase)()
            logger = logging.getLogger(phase)
            total = len(loader)
            for batch_id, (batch_x, batch_y) in enumerate(loader):
                iter_num = iter_num + 1
                batch_x, batch_y = batch_x.cuda(async=True), batch_y.cuda(async=True)
                with torch.set_grad_enabled(phase == 'train'):
                    output, vout, mu, logvar, df, MI_pred_mean, MI_log_scale = model(batch_x[:,:])
                    output_mono, vout_mono, mu_mono, logvar_mono, df_mono, MI_pred_mean, MI_log_scale = model_mono(batch_x[:,3:4])
                    loss_dict = losses['dice_ACN'](cfg, output, batch_x[:,:], batch_y, vout, mu, logvar, output_mono, batch_x[:,3:4], vout_mono, mu_mono, logvar_mono, epoch)
                    
                    #### MI part ####
                    loss_MI = {}
                    MI_loss = Cal_MIloss(df, df_mono, MI_pred_mean, MI_log_scale)
                    loss_MI['MILoss'] = MI_loss
                    loss_dict.update(loss_MI)
                    
                    # DISCRIMINATOR NETWORK
                    # seg maps, i.e. output, level
                    
                    d_main.train()
                    d_feature.train()
                    #d_main.to(device)
                    # discriminators' optimizers
                    optimizer_d_main = optim.Adam(d_main.parameters(), lr = cfg.SOLVER.LEARNING_RATE, betas=(0.9, 0.99))
                    optimizer_d_feature = optim.Adam(d_feature.parameters(), lr = cfg.SOLVER.LEARNING_RATE, betas=(0.9, 0.99))
                    # labels for EnA adversarial training
                    source_label = 0
                    target_label = 1
                    # labels for KnA adversarial training
                    df_source_label = 0
                    df_target_label = 1
                    
                    optimizer.zero_grad()
                    optimizer_d_main.zero_grad()
                    optimizer_d_feature.zero_grad()
                    
                    # only train. Don't accumulate grads in disciminators
                    for param in d_main.parameters():
                        param.requires_grad = False
                    for param in d_feature.parameters():
                        param.requires_grad = False  
                    
                    if phase == 'train':
                        (loss_dict['loss_Co'] + cfg.LAMBDA_MI * loss_dict['MILoss']).backward(retain_graph=True)
                    
                    
                    ##################### adversarial training ot fool the discriminator ######################
                    # UDA
                    pred_src_main = output
                    pred_trg_main = output_mono
                    #pred_trg_main = interp_target(pred_trg_main)
                    d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
                    loss_adv_trg_main = bce_loss(d_out_main, source_label)
                    loss = cfg.LAMBDA_ADV_EN * loss_adv_trg_main
                    loss_1 = loss
                    #loss_1.backward()
                    # DF 
                    df_src_main = df
                    df_trg_main = df_mono
                    d_df_out_main = d_feature(df_trg_main)
                    loss_adv_df_trg_main = bce_loss(d_df_out_main, df_source_label)
                    loss = cfg.LAMBDA_ADV_KN * loss_adv_df_trg_main
                    loss_df_1 = loss
                    
                    loss_1_sum = loss_1 + loss_df_1
                    if phase == 'train':
                        loss_1_sum.backward()                    
                    
                    
                    ####################### Train discriminator networks ######################################
                    # enable training mode on discriminator networks
                    for param in d_main.parameters():
                        param.requires_grad = True
                    for param in d_feature.parameters():
                        param.requires_grad = True
                    # UDA
                    # train with source
                    pred_src_main = pred_src_main.detach()
                    d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
                    loss_d_main = bce_loss(d_out_main, source_label)
                    loss_d_main_1 = loss_d_main / 2
                    #loss_d_main_1.backward()
                    
                    # DF
                    # train with source
                    df_src_main = df_src_main.detach()
                    d_df_out_main = d_feature(df_src_main)
                    loss_d_feature_main = bce_loss(d_df_out_main, df_source_label)
                    loss_d_feature_main_1 = loss_d_feature_main / 2
                    
                    loss_d_main_1_sum = loss_d_main_1 + loss_d_feature_main_1
                    if phase == 'train':
                        loss_d_main_1_sum.backward()
                    
                    ####################### train with target ##################################################
                    # UDA
                    pred_trg_main = pred_trg_main.detach()
                    d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
                    loss_d_main = bce_loss(d_out_main, target_label)
                    loss_d_main_2 = loss_d_main / 2
                    # DF
                    df_trg_main = df_trg_main.detach()
                    d_df_out_main = d_feature(df_trg_main)
                    loss_d_feature_main = bce_loss(d_df_out_main, df_target_label)
                    loss_d_feature_main_2 = loss_d_feature_main / 2
                    
                    loss_d_main_2_sum = loss_d_main_2 + loss_d_feature_main_2
                    if phase == 'train':
                        loss_d_main_2_sum.backward()
                    #loss_d_main_2.backward()
                    #print("loss_1:", loss_1, "loss_d_main_1:", loss_d_main_1, "loss_d_main_2:", loss_d_main_2)
                    
                meters.update(**loss_dict)
                
                num_classes = 4

                if phase == 'train':
                    optimizer.step()
                    optimizer_d_main.step()
                    optimizer_d_feature.step()
                    
                else:
                    if metrics and (epoch + 1) % 20 == 0:
                        with torch.no_grad():
                            hausdorff = metrics['hd']
                            metric_dict = hausdorff(output, batch_y)
                            meters.update(**metric_dict)
                        save_sample(output, output_mono, batch_x, batch_y, epoch, batch_id)
                logger.info(meters.delimiter.join([f"Epoch: {epoch}, Batch:{batch_id}/{total}",
                                                   f"{str(meters)}",
                                                   f"Time: {time.time() - end: .3f}"
                                                   ]))
                end = time.time()

            if phase == 'eval':
                dice = 1 - (meters.ed_mono_dc_loss.global_avg + meters.net_mono_dc_loss.global_avg + meters.et_mono_dc_loss.global_avg) / 3
                state = {}
                state['model'] = model.state_dict()
                state['model_mono'] = model_mono.state_dict()
                state['d_main'] = d_main.state_dict()
                state['d_feature'] = d_feature.state_dict()
                state['optimizer'] = optimizer.state_dict()
                file_name = os.path.join(cfg.LOG_DIR, cfg.TASK_NAME, 'epoch' + str(epoch) + '.pt')
                torch.save(state, file_name)
                if dice > best_dice:
                    best_dice = dice
                    shutil.copyfile(file_name, os.path.join(cfg.LOG_DIR, cfg.TASK_NAME, 'best_model.pth'))

    return model

def main():
    init_env('0')
    loaders = make_data_loaders(cfg)
    model, model_mono = build_model(cfg)
    model = model.cuda()
    ##mono_model
    model_mono = model_mono.cuda()
    ## adv model
    d_main = get_fc_discriminator(num_classes = 4).cuda()
    d_feature = get_df_discriminator(num_classes = 128).cuda()
    task_name = 'Task_brats18_onlyT1ce_test2'
    log_dir = os.path.join(cfg.LOG_DIR, task_name)
    cfg.TASK_NAME = task_name
    
    optimizer, scheduler = make_optimizer_double(cfg, model, model_mono)
    metrics = get_metrics(cfg)
    losses = get_losses(cfg)
    
    continue_training = False
    epoch = 0
        
    mkdir(log_dir)
    logger = setup_logger('train', log_dir, filename='train.log')
    logger.info(cfg)
    logger = setup_logger('eval', log_dir, filename='eval.log')
    train_val(model, model_mono, d_main, d_feature, loaders, optimizer, scheduler, losses, metrics, epoch)
    
    
        
if __name__ == "__main__":
    main()