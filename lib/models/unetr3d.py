from matplotlib.pyplot import grid
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from timm.models.layers.helpers import to_3tuple
from lib.models.mae3d import build_3d_sincos_position_embedding

import time

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=False):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.in_chans = in_chans
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten
        self.in_chan_last = in_chan_last

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        # pdb.set_trace()
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x

class PatchEmbed2P1D(nn.Module):
    """ 2D + 1D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=False):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.in_chans = in_chans
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten
        self.in_chan_last = in_chan_last

        kernel_size1 = (patch_size[0], patch_size[1], 1)
        kernel_size2 = (1, 1, patch_size[2])

        self.proj = nn.Sequential(nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size1, stride=kernel_size1),
                                  nn.Conv3d(embed_dim, embed_dim, kernel_size=kernel_size2, stride=kernel_size2))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        # pdb.set_trace()
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x

class UNETR3D(nn.Module):
    """General segmenter module for 3D medical images
    """
    def __init__(self, encoder, decoder, args):
        super().__init__()
        if args.spatial_dim == 3:
            input_size = (args.roi_x, args.roi_y, args.roi_z)
        elif args.spatial_dim == 2:
            input_size = (args.roi_x, args.roi_y)

        self.encoder = encoder(img_size=input_size,
                               patch_size=args.patch_size,
                               in_chans=args.in_chans,
                               embed_dim=args.encoder_embed_dim,
                               depth=args.encoder_depth,
                               num_heads=args.encoder_num_heads,
                               drop_path_rate=args.drop_path,
                               embed_layer=PatchEmbed3D,
                               use_learnable_pos_emb=True,
                               return_hidden_states= True
                               )
        self.decoder = decoder(in_channels=args.in_chans,
                               out_channels=args.num_classes,
                               img_size=input_size,
                               patch_size=args.patch_size,
                               feature_size=args.feature_size,
                               hidden_size=args.encoder_embed_dim,
                               spatial_dims=args.spatial_dim)
    
    def get_num_layers(self):
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def no_weight_decay(self):
        total_set = set()
        module_prefix_dict = {self.encoder: 'encoder',
                              self.decoder: 'decoder'}
        for module, prefix in module_prefix_dict.items():
            if hasattr(module, 'no_weight_decay'):
                for name in module.no_weight_decay():
                    total_set.add(f'{prefix}.{name}')
        print(f"{total_set} will skip weight decay")
        return total_set
    
    def forward(self, x_in, time_meters=None):
        """
        x_in in shape of [BCHWD]
        """
        s_time = time.perf_counter()
        x, hidden_states = self.encoder(x_in, time_meters=time_meters)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['enc'].append(time.perf_counter() - s_time)

        s_time = time.perf_counter()
        logits = self.decoder(x_in, x, hidden_states)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['dec'].append(time.perf_counter() - s_time)
        return logits
