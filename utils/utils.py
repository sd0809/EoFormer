import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import torch.distributed as dist
import warnings
from medpy.metric import binary

    
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
 
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(
            len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas
 
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank *
                          self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)
 
    def __len__(self):
        return self.num_samples
        

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


# def calculate_metric(y_pred=None, y=None, eps=1e-9):
    
#     if y.shape != y_pred.shape:
#         raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")
    
#     batch_size, n_class = y_pred.shape[:2]
    
#     dsc = np.empty((batch_size, n_class))
#     hd = np.empty((batch_size, n_class))
#     cnt = np.zeros((n_class))
#     for b, c in np.ndindex(batch_size, n_class):
#         edges_pred, edges_gt = y_pred[b, c], y[b, c]
#         if not np.any(edges_gt):
#             warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan distance.")
#         if not np.any(edges_pred):
#             warnings.warn(f"the prediction of class {c} is all 0, this may result in nan distance.")
        
#         if  (edges_pred.sum()>0 and edges_gt.sum()>0): # pred和gt均不为0，正常计算dice和hausdorff
#             dice = binary.dc(edges_pred, edges_gt)
#             distance = binary.hd95(edges_pred, edges_gt)
#             dsc[b, c] = dice
#             hd[b, c] = distance
#             cnt[c] += 1
#         elif  (edges_pred.sum()==0 and edges_pred.sum() == 0):  # pred和gt均为0，dice=1 hausdorff 0
#             dsc[b, c] = 1
#             hd[b, c] = 0
#             cnt[c] += 1
#         # elif  ( (edges_pred.sum()>0 and edges_gt.sum()==0) or (edges_pred.sum()==0 and edges_gt.sum()>0) ): # pred和gt中有一个为0，dice=0 hausdorff 373.128664
#         else:
#             dsc[b, c] = 0
#             hd[b, c] = 0
#             cnt[c] += 1
#     dsc = np.sum(dsc, axis=0)
#     hd = np.sum(hd, axis=0)
#     dsc = dsc / cnt
#     hd = hd / cnt
    
#     return torch.from_numpy(dsc), torch.from_numpy(hd)


def calculate_metric(y_pred=None, y=None, eps=1e-6):
    
    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")
    
    batch_size, n_class = y_pred.shape[:2]
    
    dsc = np.empty((batch_size, n_class))
    hd = np.empty((batch_size, n_class))
    cnt = np.zeros((n_class))
    for b, c in np.ndindex(batch_size, n_class):
        edges_pred, edges_gt = y_pred[b, c], y[b, c]
        if not np.any(edges_gt):
            warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan distance.")
        if not np.any(edges_pred):
            warnings.warn(f"the prediction of class {c} is all 0, this may result in nan distance.")
        
        if  (edges_pred.sum()>0 and edges_gt.sum()>0): # pred和gt均不为0，正常计算dice和hausdorff
            dice = binary.dc(edges_pred, edges_gt)
            distance = binary.hd95(edges_pred, edges_gt)
            dsc[b, c] = dice
            hd[b, c] = distance
            cnt[c] += 1
        elif  (edges_pred.sum()==0 and edges_pred.sum() == 0):  # pred和gt均为0，dice=1 hausdorff 0
            dsc[b, c] = 1
            hd[b, c] = 0
            cnt[c] += 1
        # elif  ( (edges_pred.sum()>0 and edges_gt.sum()==0) or (edges_pred.sum()==0 and edges_gt.sum()>0) ): # pred和gt中有一个为0，dice=0 hausdorff 373.128664
        else:
            dsc[b, c] = 0
            hd[b, c] = 0
            cnt[c] += eps
    dsc = np.sum(dsc, axis=0)
    hd = np.sum(hd, axis=0)
    dsc = dsc / cnt
    hd = hd / cnt
    
    return torch.from_numpy(dsc), torch.from_numpy(hd)