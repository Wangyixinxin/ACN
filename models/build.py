#!/usr/bin/env python3
# encoding: utf-8
from .unet import UNet3D, UnetVAE3D


def build_model(cfg):
    if cfg.MODEL.NAME == 'unet-vae':
        model = UnetVAE3D(cfg.DATASET.INPUT_SHAPE,
                          in_channels=len(cfg.DATASET.USE_MODES),
                          out_channels=4,
                          init_channels=cfg.MODEL.INIT_CHANNELS,
                          p=cfg.MODEL.DROPOUT)
    elif cfg.MODEL.NAME == 'ACN-unet':
        model = UnetVAE3D(cfg.DATASET.INPUT_SHAPE,
                          in_channels=4,
                          out_channels=4,
                          init_channels=cfg.MODEL.INIT_CHANNELS,
                          p=cfg.MODEL.DROPOUT)
        
        model_mono = UnetVAE3D(cfg.DATASET.INPUT_SHAPE,
                          in_channels=1,
                          out_channels=4,
                          init_channels=cfg.MODEL.INIT_CHANNELS,
                          p=cfg.MODEL.DROPOUT)
        return model, model_mono
    elif cfg.MODEL.NAME == 'unet-vae-UDA-Distill':
        model = UnetVAE3D(cfg.DATASET.INPUT_SHAPE,
                          in_channels=4,
                          out_channels=4,
                          init_channels=cfg.MODEL.INIT_CHANNELS,
                          p=cfg.MODEL.DROPOUT)
        
        model_mono = UnetVAE3D(cfg.DATASET.INPUT_SHAPE,
                          in_channels=1,
                          out_channels=4,
                          init_channels=cfg.MODEL.INIT_Distill_CHANNELS,
                          p=cfg.MODEL.DROPOUT)
        return model, model_mono
    else:
        model = UNet3D(cfg.DATASET.INPUT_SHAPE,
                       in_channels=len(cfg.DATASET.USE_MODES),
                       out_channels=4,
                       init_channels=cfg.MODEL.INIT_CHANNELS,
                       p=cfg.MODEL.DROPOUT)

    return model
