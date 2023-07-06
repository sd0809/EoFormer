import os
import pdb
import json
from cv2 import log
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator, List, Optional, Union
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

# 混合精度训练
from torch.cuda.amp import autocast as autocast
from torch.cuda import amp

from utils.utils import calculate_metric

cal_mean_dice= DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True) # 返回一批数据的TC WT ET dice平均值, 若nan 返回0
cal_hausdorff = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, percentile=95, get_not_nans=True) # 我们的0类不是背景，所以要包含bg(默认为第0维度)进去计算

post_sigmoid = Activations(sigmoid=True)
scaler = amp.GradScaler(enabled=True)


def train(model, optimizer, loss_fn, data_loader, device):
    model.train()
    n_ctr = 0
    n_loss = 0
    for step, batch_data in enumerate(data_loader):
        optimizer.zero_grad()
        images, labels = batch_data['image'], batch_data['label']
        images = images.to(device)
        labels = labels.to(device, dtype=torch.float32)
        with autocast(enabled=True):
            probs = model(images) # shape: [bs, n_classes, 24, 256, 256]
            loss = loss_fn(probs, labels)

        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        # for name, param in model.named_parameters():
        #     if param.grad == None:
        #         print(name)
        scaler.step(optimizer)
        scaler.update()
        n_loss += loss.item()
        n_ctr += 1
    return n_loss/n_ctr


def evaluate(args, model, model_inferer, loss_fn, data_loader, device='cuda'):
    
    model.eval()
    with torch.no_grad():
        n_ctr = 0
        n_ctr_et = 0
        n_loss = 0
        n_dice_wt = 0
        n_dice_tc = 0
        n_dice_et = 0
        n_hausdorff_wt = 0
        n_hausdorff_tc = 0
        n_hausdorff_et = 0
        for step, batch_data in enumerate(data_loader):
            # if step == 1:
            #     break
            images, labels = batch_data['image'], batch_data['label']
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)
            with autocast(enabled=True):
                probs = model_inferer(images)
                loss = loss_fn(probs, labels) # 分类损失函数
            n_loss += loss.item()
            
            probs_sigmoid = post_sigmoid(probs)
            pred_masks = (probs_sigmoid >= 0.5).to(device, dtype=torch.float32)
        
            # cal_mean_dice.reset()
            # cal_hausdorff.reset()
            # dce = cal_mean_dice(pred_masks, labels)
            # haus = cal_hausdorff(pred_masks, labels)
            # dice, dice_not_nans = cal_mean_dice.aggregate() # 去除掉结果为nan的项
            # hausdorff, hausdorff_not_nans = cal_hausdorff.aggregate() # 去除掉结果为nan的项
            
            dice , hausdorff = calculate_metric(pred_masks, labels)
            
            n_dice_tc += dice[0].item()
            n_dice_wt += dice[1].item()
            n_dice_et += dice[2].item()

            n_hausdorff_tc += hausdorff[0].item()
            n_hausdorff_wt += hausdorff[1].item()
            n_hausdorff_et += hausdorff[2].item()
            
            if hausdorff[2].item() > 1e-3:
                n_ctr_et += 1
            elif hausdorff[2].item() == 0.0 and dice[2].item() == 1.0:
                n_ctr_et += 1
            else:
                n_ctr_et += 0
            n_ctr += 1
            
            dice_wt = n_dice_wt/n_ctr
            dice_tc = n_dice_tc/n_ctr
            dice_et = n_dice_et/n_ctr_et
            hausdorff_wt = n_hausdorff_wt/n_ctr
            hausdorff_tc = n_hausdorff_tc/n_ctr
            hausdorff_et = n_hausdorff_et/n_ctr_et

    return n_loss/n_ctr, (dice_wt + dice_tc + dice_et)/3, dice_wt, dice_tc, dice_et, (hausdorff_wt + hausdorff_tc + hausdorff_et)/3, hausdorff_wt, hausdorff_tc, hausdorff_et


def test(model, model_inferer, data_loader, saver0, saver1, device='cuda'):
    
    model.eval()
    with torch.no_grad():
        n_ctr = 0
        n_ctr_et = 0
        n_dice_wt = 0
        n_dice_tc = 0
        n_dice_et = 0
        n_hausdorff_wt = 0
        n_hausdorff_tc = 0
        n_hausdorff_et = 0
        for step, batch_data in enumerate(data_loader):
            images, labels = batch_data['image'], batch_data['label']
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)
            
            with autocast(enabled=True):
                probs = model_inferer(images)
            
            probs_sigmoid = post_sigmoid(probs)
            pred_masks = (probs_sigmoid >= 0.5).to(device, dtype=torch.float32)
            
            label = labels[0, 1]
            # label[np.where(labels[0, 1] == 1)] = 1
            label[np.where(labels[0, 0] == 1)] = 2
            label[np.where(labels[0, 2] == 1)] = 3

            seg_img = pred_masks[0, 1]
            # seg_img[np.where(pred_masks[0, 1] == 1)] = 1
            seg_img[np.where(pred_masks[0, 0] == 1)] = 2
            seg_img[np.where(pred_masks[0, 2] == 1)] = 3
            
            saver0(label)
            saver1(seg_img)
            
            dice , hausdorff = calculate_metric(pred_masks, labels)
            
            print('dice:')
            print('tc:', dice[0].item())
            print('wt:', dice[1].item())
            print('et:', dice[2].item())
            print('hausdorff:')
            print('tc:', hausdorff[0].item())
            print('wt:', hausdorff[1].item())
            print('et:', hausdorff[2].item())
            
            n_dice_tc += dice[0].item()
            n_dice_wt += dice[1].item()
            n_dice_et += dice[2].item()

            n_hausdorff_tc += hausdorff[0].item()
            n_hausdorff_wt += hausdorff[1].item()
            n_hausdorff_et += hausdorff[2].item()
            
            if hausdorff[2].item() > 1e-3:
                n_ctr_et += 1
            elif hausdorff[2].item() == 0.0 and dice[2].item() == 1.0:
                n_ctr_et += 1
            else:
                n_ctr_et += 0
            n_ctr += 1
        
        dice_wt = n_dice_wt/n_ctr
        dice_tc = n_dice_tc/n_ctr
        dice_et = n_dice_et/n_ctr_et
        hausdorff_wt = n_hausdorff_wt/n_ctr
        hausdorff_tc = n_hausdorff_tc/n_ctr
        hausdorff_et = n_hausdorff_et/n_ctr_et
    return (dice_wt + dice_tc + dice_et)/3, dice_wt, dice_tc, dice_et, (hausdorff_wt + hausdorff_tc + hausdorff_et)/3, hausdorff_wt, hausdorff_tc, hausdorff_et


def inference(model, model_inferer, data_loader, saver, device='cuda'):
    
    model.eval()
    with torch.no_grad():
        n_ctr = 0
        
        for step, batch_data in enumerate(data_loader):
            images = batch_data['image']
            images = images.to(device)
            
            with autocast(enabled=True):
                probs = model_inferer(images)
            
            probs_sigmoid = post_sigmoid(probs)
            pred_masks = (probs_sigmoid >= 0.5).to(device, dtype=torch.float32)
            
            seg_img = pred_masks[0, 1]
            # seg_img[np.where(pred_masks[0, 1] == 1)] = 1
            seg_img[np.where(pred_masks[0, 0] == 1)] = 2
            seg_img[np.where(pred_masks[0, 2] == 1)] = 3
            saver(seg_img)
            n_ctr += 1
            
    return n_ctr