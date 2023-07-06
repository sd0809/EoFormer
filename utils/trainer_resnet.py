import os 
from cv2 import log
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
from itertools import cycle
from scipy import interp
from tqdm import tqdm
from typing import Iterator, List, Optional, Union
# from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize 
from torch.autograd import Variable
from sklearn.metrics import f1_score
from utils.ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix,roc_curve, auc,roc_auc_score    # 生成混淆矩阵的函数
from operator import itemgetter
from torch.cuda.amp import autocast

# from monai.metrics import compute_meandice, compute_hausdorff_distance
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from monai.transforms import AsChannelLast
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
import torch.nn.functional as F

from scipy import ndimage

cal_mean_dice = DiceMetric()
cal_hausdorff = HausdorffDistanceMetric()


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)

    
def train(model, optimizer, loss_fn, data_loader, device):
    model.train()
    n_ctr = 0
    n_loss = 0
    n_dice = 0
    n_hausdorff = 0
    # data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        # 梯度清零
        optimizer.zero_grad()
        IDs, images, labels, label_masks = data
        print(IDs[0])
        # print(images.shape)
        images = images.to(device, dtype=torch.float32)
        label_masks = label_masks.to(device, dtype=torch.int64)  # CE loss 输入需要 long type data
        label_masks = expand_as_one_hot(label_masks, 2) # 对mask 做one-hot 编码
        if ((not np.any(label_masks[0][0].cpu().numpy())) or (not np.any(label_masks[0][1].cpu().numpy()))):
            print(IDs[0])
            print('background has 1 ?:', np.any(label_masks[0][0].cpu().numpy()))
            print('background: \n' , label_masks[0][0].cpu().numpy())
            print('foreground has 1 ?:', np.any(label_masks[0][1].cpu().numpy()))
            print('foreground: \n', label_masks[0][1].cpu().numpy())
            exit(0)
        probs = model(images)

        # 把mask resize到output的size
        [bs, c, d, h, w] = probs.shape  # [bs, 2, 182, 218, 182]  # 插帧有问题
        new_label_masks = np.zeros([bs, c, d, h, w])
        for label_id in range(bs):
            label_mask = np.array(label_masks[label_id].cpu(), dtype=np.float32)
            [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
            scale = [1, d*1.0/ori_d, h*1.0/ori_h, w*1.0/ori_w]
            label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
            label_mask = np.argmax(label_mask, axis=0)
            new_label_masks[label_id] = label_mask

        new_label_masks = torch.tensor(new_label_masks, dtype=torch.float32).cuda()  # 会不会全0? 不会
        if ((not np.any(new_label_masks[0][0].cpu().numpy())) or (not np.any(new_label_masks[0][1].cpu().numpy()))):
            print(IDs[0])
            print('background has 1 ?:', np.any(new_label_masks[0][0].cpu().numpy()))
            print('background: \n' , new_label_masks[0][0].cpu().numpy())
            print('foreground has 1 ?:', np.any(new_label_masks[0][1].cpu().numpy()))
            print('foreground: \n', new_label_masks[0][1].cpu().numpy())
            exit(0)

        loss = loss_fn(probs, new_label_masks) # 分类损失函数
        loss.backward()
        optimizer.step()

        pred_masks = F.softmax(probs, dim=1)  # 按行计算
        pred_masks = torch.argmax(probs, dim=1)  # 二值化
        pred_masks = expand_as_one_hot(pred_masks, 2)  # 对pred_mask 做one-hot 编码

        n_loss += loss.item()
        n_dice += cal_mean_dice(pred_masks, new_label_masks).mean().item()
        n_hausdorff += cal_hausdorff(pred_masks, new_label_masks).mean().item()
        n_ctr += 1

    return n_loss/n_ctr, n_dice/n_ctr, n_hausdorff/n_ctr


def evaluate(model, loss_fn, data_loader, device='cuda'):
    
    model.eval()
    with torch.no_grad():

        n_ctr = 0
        n_loss = 0
        n_dice = 0
        n_hausdorff = 0
        # data_loader = tqdm(data_loader)
        for step, data in enumerate(data_loader):
            IDs, images, labels, label_masks = data
            
            images = images.to(device, dtype=torch.float32)
            label_masks = label_masks.to(device, dtype=torch.int64)
            label_masks = expand_as_one_hot(label_masks, 2) # 对mask 做one-hot 编码

            probs = model(images)
            probs = F.softmax(probs, dim=1)
            
            # 把输出resize到原始mask的size
            [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
            [channel, depth, height, width] = label_masks[0].shape
            pred_masks=[]
            for mask_idx in range(batchsize):
                pred_mask = probs[mask_idx].cpu()
                pred_mask = np.array(pred_mask, dtype=np.float32)  # float32才能做zoom
                scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
                pred_mask = ndimage.interpolation.zoom(pred_mask, scale, order=1)
                # pred_mask = np.argmax(pred_mask, axis=0)
                pred_mask = torch.tensor(pred_mask, dtype=torch.int64)
                pred_mask = torch.argmax(pred_mask, dim=0)
                pred_masks.append(pred_mask)
            
            pred_masks = torch.tensor(np.stack(pred_masks, axis=0)).to(device, dtype=torch.int64)
            pred_masks = expand_as_one_hot(pred_masks, 2)  # 对pred_mask 做one-hot 编码
            n_ctr += 1
            n_dice += cal_mean_dice(pred_masks, label_masks).mean().item()
            n_hausdorff += cal_hausdorff(pred_masks, label_masks).mean().item()
            # print('ID:', IDs)
            # print('dice:', cal_mean_dice(pred_masks, label_masks))
            # print('hausdorff:', cal_hausdorff(pred_masks, label_masks))
            # print(' - - -' * 15)

    return n_dice/n_ctr, n_hausdorff/n_ctr


def test_and_show(model, loss_fn, data_loader, save_folder, device='cuda'):
    
    model.eval()
    bad_res_subjects = []
    with torch.no_grad():
        
        num_ctr = 0
        n_dice = 0
        n_hausdorff = 0
        data_loader = tqdm(data_loader)
        for step, data in enumerate(data_loader):

            IDs, images, labels, label_masks = data
            # print('ID:', IDs)
            images = images.to(device, dtype=torch.float32) # float16 train会导致nan
            labels = labels.to(device)
            label_masks = label_masks.to(device, dtype=torch.int32)
            label_masks = expand_as_one_hot(label_masks, 2) # 对mask 做one-hot 编码
            probs = model(images)
            probs = F.softmax(probs, dim=1)
            
            # 把输出resize到原始mask的size
            [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
            [channel, depth, height, width] = label_masks[0].shape
            for mask_idx in range(batchsize):
                sub_save_folder = save_folder + f'/{IDs[mask_idx]}'
                num_ctr += 1
                pred_mask = probs[mask_idx].cpu()
                label_mask = label_masks[mask_idx]
                pred_mask = np.array(pred_mask, dtype=np.float32)  # float32才能做zoom
                scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
                pred_mask = ndimage.interpolation.zoom(pred_mask, scale, order=1)
                pred_mask = np.argmax(pred_mask, axis=0)
                pred_mask = expand_as_one_hot(pred_mask, 2)  # 对pred_mask 做one-hot 编码
                save_array(IDs[mask_idx], pred_mask, sub_save_folder)
                pred_mask = torch.tensor(pred_mask).to(device, dtype=torch.int32)
                dice = cal_mean_dice(pred_mask, label_mask).mean().item()
                hausdorff = cal_hausdorff(pred_mask, label_mask).mean().item()
                n_dice += dice
                n_hausdorff += hausdorff
                if dice < 0.70:
                    bad_res_subjects.append(IDs[mask_idx])
    return n_dice/num_ctr, n_hausdorff/num_ctr, bad_res_subjects


def set_require_grad(model, required_grad_lst):
    if 'dense' not in required_grad_lst:
        print('dense layer does not require grad!')
        for param_name, param in model.named_parameters():
            if param_name.startswith("module.dense."):
                param.requires_grad = False
    if 'layer4' not in required_grad_lst:
        print('layer4 does not require grad!')
        for param_name, param in model.named_parameters():
            if param_name.startswith("module.back_bone.layer4."):
                param.requires_grad = False
    if 'layer3' not in required_grad_lst:
        print('layer3 does not require grad!')
        for param_name, param in model.named_parameters():
            if param_name.startswith("module.back_bone.layer3."):
                param.requires_grad = False
    if 'layer2' not in required_grad_lst:
        print('layer2 does not require grad!')
        for param_name, param in model.named_parameters():
            if param_name.startswith("module.back_bone.layer2."):
                param.requires_grad = False
    if 'layer1' not in required_grad_lst:
        print('layer1 does not require grad!')
        for param_name, param in model.named_parameters():
            if param_name.startswith("module.back_bone.layer1."):
                param.requires_grad = False


def save_train_history(path, model_name, tags, value):
    """writing model weights and training logs to files."""

    log_names = tags
    pd.DataFrame(
        dict(zip(log_names, value)), index=[0]
    ).to_csv(f"{path}/train_log_{model_name}.csv", mode='a', index=False)

def save_array(ID, pred_mask,save_folder):
    np.save(f'{save_folder}/{ID}.npy', pred_mask)
    return