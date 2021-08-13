#!/usr/bin/env python3
# encoding: utf-8
from yacs.config import CfgNode as CN
import platform

_C = CN()
_C.DATASET = CN()

if "Win" in platform.system():
    _C.DATASET.DATA_ROOT = './data/MICCAI_BraTS_2018_Data_Training'
else:
    _C.DATASET.DATA_ROOT = "./data/MICCAI_BraTS_2018_Data_Training"

_C.DATASET.NUM_FOLDS = 3
_C.DATASET.SELECT_FOLD = 0
_C.DATASET.USE_MODES = ("t1", "t2", "flair", "t1ce")

_C.DATASET.INPUT_SHAPE = (160, 192, 128)

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 1
_C.DATALOADER.NUM_WORKERS = 6

_C.MODEL = CN()
_C.MODEL.NAME = 'ACN-unet'
_C.MODEL.INIT_CHANNELS = 16
_C.MODEL.INIT_Distill_CHANNELS = 8
_C.MODEL.DROPOUT = 0.2
_C.MODEL.LOSS_WEIGHT = 0.1
_C.MODEL.LOSS_WEIGHT_MONO = 0.8

_C.SOLVER = CN()
_C.SOLVER.LEARNING_RATE = 1e-4
_C.SOLVER.WEIGHT_DECAY = 1e-5
_C.SOLVER.POWER = 0.9
_C.SOLVER.NUM_EPOCHS = 300

_C.consistency_type = 'mse'

# Adversarial training params
_C.LEARNING_RATE_D = 1e-4
_C.LAMBDA_ADV_EN = 0.001
_C.LAMBDA_ADV_KN = 0.0002

_C.LAMBDA_ADV_DF = 0.05
_C.LAMBDA_MI = 0.5

_C.MISC = CN()
_C.LOG_DIR = './logs'
