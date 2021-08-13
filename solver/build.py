#!/usr/bin/env python3
# encoding: utf-8
import torch
from .scheduler import PolyLR

def make_optimizer(cfg, model):
    lr = cfg.SOLVER.LEARNING_RATE
    print('initial learning rate is ', lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = PolyLR(optimizer, max_epoch=cfg.SOLVER.NUM_EPOCHS, power=cfg.SOLVER.POWER)

    return optimizer, scheduler

def make_optimizer_double(cfg, model1, model2):
    lr = cfg.SOLVER.LEARNING_RATE
    print('initial learning rate is ', lr)
    optimizer = torch.optim.Adam([
    {'params': model1.parameters()},
    {'params': model2.parameters()}], lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = PolyLR(optimizer, max_epoch=cfg.SOLVER.NUM_EPOCHS, power=cfg.SOLVER.POWER)

    return optimizer, scheduler