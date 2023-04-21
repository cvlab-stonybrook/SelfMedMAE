import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import Block
from lib.networks.patch_embed_layers import PatchEmbed2D

__all__ = [
    'build_2d_sincos_position_embedding',
    'MAEViTEncoder', 
    'MAEViTDecoder',
    'mae_encoder_small_patch16_224',
    'mae_decoder_small_patch16_224'
]

def build_2d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    """
    TODO: the code can work when grid size is isotropic (H==W), but it is not logically right especially when data is non-isotropic(H!=W).
    """
    h, w = grid_size, grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


class MAEViTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Modified from timm implementation
    """
    def __init__(self, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed2D, norm_layer=None, act_layer=None, use_pe=True, return_patchembed=False):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 # don't consider distillation here
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.use_pe = use_pe
        self.return_patchembed = return_patchembed

        self.patch_embed = embed_layer(img_size=patch_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        assert self.patch_embed.num_patches == 1, \
                "Current embed layer should output 1 token because the patch length is reshaped to batch dimension"

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_pe = nn.Parameter(torch.zeros([1, 1, embed_dim], dtype=torch.float32))
        # self.cls_pe.requires_grad = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # init patch embed parameters
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_token, std=.02, a=-.02, b=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward_features(self, x, pos_embed=None):
        return_patchembed = self.return_patchembed

        embed_dim = self.embed_dim
        B, L, _ = x.shape

        x = self.patch_embed(x) # [B*L, embed_dim]
        x = x.reshape(B, L, embed_dim)
        if return_patchembed:
            patchembed = x
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        if self.use_pe:
            if x.size(1) != pos_embed.size(1):
                assert x.size(1) == pos_embed.size(1) + 1, "Unmatched x and pe shapes"
                cls_pe = torch.zeros([B, 1, embed_dim], dtype=torch.float32).to(x.device)
                pos_embed = torch.cat([cls_pe, pos_embed], dim=1)
            x = self.pos_drop(x + pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if return_patchembed:
            return x, patchembed
        else:
            return x

    def forward(self, x, pos_embed=None):
        if self.return_patchembed:
            x, patch_embed = self.forward_features(x, pos_embed)
        else:
            x = self.forward_features(x, pos_embed)
        x = self.head(x)
        if self.return_patchembed:
            return x, patch_embed
        else:
            return x

class MAEViTDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Modified from timm implementation
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=None, act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * patch_size ** 2
        self.embed_dim = embed_dim
        self.num_tokens = 1 # don't consider distillation here
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def mae_encoder_small_patch16_224(**kwargs):
    model = MAEViTEncoder(
        embed_dim=384, 
        num_heads=6,
        **kwargs)
    return model

def mae_decoder_small_patch16_224(**kwargs):
    model = MAEViTDecoder(
        embed_dim=128, 
        depth=4,
        num_heads=3,
        **kwargs)
    return model
