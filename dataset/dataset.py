import os
import json
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from monai.data import CacheDataset
from monai.transforms import Resize, MapTransform

def split_data(datalist, basedir):

    with open(datalist) as f:
        json_data = json.load(f)

    train_data = json_data['train']
    valid_data = json_data['valid']
    test_data = json_data['test']
    
    train = []
    valid = []
    test = []
    for d in train_data:
        for k, v in d.items():  
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
        train.append(d)

    for d in valid_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
        valid.append(d)
    
    for d in test_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
        test.append(d)
    return train, valid, test


def get_dataset_brats(data_path: str, 
                  json_file: str, 
                  transform=None,
                 )-> Tuple[Dataset, Dataset]:
    
    train_list, valid_list, test_list = split_data(json_file, data_path)
    
    train_set = CacheDataset(
                            data = train_list,
                            cache_rate=0.0,
                            transform = transform['train']
                            )
    valid_set = CacheDataset(
                            data = valid_list,
                            cache_rate=0.0,
                            transform = transform['valid']
                            )
    test_set = CacheDataset(
                            data = test_list,
                            cache_rate=0.0,
                            transform = transform['valid']
                            )
    
    return train_set, valid_set, test_set, train_list, valid_list, test_list