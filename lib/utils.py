import os
import sys

import random
import builtins
import warnings
import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import argparse
from omegaconf import OmegaConf
from collections import deque
import importlib

from sklearn.metrics import roc_auc_score

def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        # np.random.seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.benchmark = True

def dist_setup(ngpus_per_node, args):
    torch.multiprocessing.set_start_method('fork', force=True)
    # suppress printing if not master
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()

def get_conf():
    # First of all, parse config file
    conf_file = sys.argv[1]
    assert os.path.exists(conf_file), f"Config file {conf_file} does not exist!"
    conf = OmegaConf.load(conf_file)

    # build argparser from config file in order to overwrite with cli options
    parser = argparse.ArgumentParser(description='SSL pre-training')
    parser.add_argument('conf_file', type=str, help='path to config file')
    for key in conf:
        if conf[key] is None:
            parser.add_argument(f'--{key}', default=None)
        else:
            if key == 'gpu':
                parser.add_argument('--gpu', type=int, default=conf[key])
            elif key == 'multiprocessing_distributed':
                parser.add_argument('--multiprocessing_distributed', type=bool, default=conf[key])
            else:
                parser.add_argument(f'--{key}', type=type(conf[key]), default=conf[key])
    args = parser.parse_args()
    if args.gpu:
        args.gpu = int(args.gpu)
    args.output_dir = '/'.join([*(args.output_dir.split('/')[:-1]), args.run_name])
    args.ckpt_dir = f"{args.output_dir}/ckpts"
    if not hasattr(args, 'num_samples'):
        args.num_samples = 4
    print(args)
    return args

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
        print(f'count: {self.count} | total: {self.total}')

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def get_vit_layer_id(var_name, num_max_layer, prefix=''):
    if var_name in (prefix + "cls_token", prefix + "mask_token", prefix + "pos_embed"):
        return 0
    elif var_name.startswith(prefix + "patch_embed"):
        return 0
    elif var_name.startswith(prefix + "rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith(prefix + "blocks"):
        names = var_name.split('.')
        anchor_ind = names.index('blocks') # 'blocks' is an anchor
        block_id = int(names[anchor_ind + 1])
        return block_id + 1
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        if layer_id is not None:
            return self.values[layer_id]
        else:
            return 1

    def get_layer_id(self, var_name, prefix=''):
        return get_vit_layer_id(var_name, len(self.values), prefix)


@torch.no_grad()
def concat_all_gather(tensor, distributed=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if distributed:
        dist.barrier()
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(dist.get_world_size())]
        # print(f"World size: {dist.get_world_size()}")
        dist.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor

def compute_aucs(pred, gt):
    auc_list = []
    if pred.ndim == 2:
        n_classes = pred.shape[1]
    elif pred.ndim == 1:
        n_classes = 1
    else:
        raise ValueError("Prediction shape wrong")
    for i in range(n_classes):
        try:
            auc = roc_auc_score(gt[:, i], pred[:, i])
        except (IndexError, ValueError) as error:
            if isinstance(error, IndexError):
                auc = roc_auc_score(gt, pred)
            elif isinstance(error, ValueError):
                auc = 0
            else:
                raise Exception("Unexpected Error")
        auc_list.append(auc)
    mAUC = np.mean(auc_list)
    return auc_list, mAUC

def masks_to_3d_boxes(masks):
    """
    Compute the 3D bounding boxes around the provided masks.

    Returns a [N, 6] tensor containing 3D bounding boxes. The boxes are in ``(x1, y1, z1, x2, y2, z2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2``.

    Args:
        masks (Tensor[N, H, W, D]): masks to transform where N is the number of masks
            and (H, W, D) are the spatial dimensions.

    Returns:
        Tensor[N, 6]: 3d bounding boxes
    """
    if not isinstance(masks, torch.Tensor):
        masks = torch.tensor(masks)
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device, dtype=torch.int8)
    
    if len(masks.shape) == 3:
        masks = masks[None, ...]
    elif len(masks.shape) != 4:
        raise Exception(f"Unsupported masks dimension {len(masks.shape)}")

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 6), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x, z = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.min(z)
        bounding_boxes[index, 3] = torch.max(x)
        bounding_boxes[index, 4] = torch.max(y)
        bounding_boxes[index, 5] = torch.max(z)

    return bounding_boxes
