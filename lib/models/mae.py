import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lib.networks.patch_embed_layers import PatchEmbed2D

from networks import build_2d_sincos_position_embedding
from timm.models.layers import trunc_normal_

import math

__all__ = ["MAE"]

def patchify_image(x: Tensor, patch_size: int = 16):
    # patchify input, [B,C,H,W] --> [B,C,gh,ph,gw,pw] --> [B,gh*gw,C*ph*pw]
    B, C, H, _ = x.shape
    grid_size = H // patch_size

    x = x.reshape(B, C, grid_size, patch_size, grid_size, patch_size) # [B,C,gh,ph,gw,pw]
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, grid_size ** 2, C * patch_size ** 2) # [B,gh*gw,C*ph*pw]

    return x

def batched_shuffle_indices(batch_size, length, device):
    """
    Generate random permutations of specified length for batch_size times
    Motivated by https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    """
    rand = torch.rand(batch_size, length).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm

class FourierMSELoss(nn.Module):
    def __init__(self, embed_dim, temperature=0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
    
    def forward(self, x, target):
        # output channels
        out_chans = x.size(-2)
        # embedding dimension and temperature
        embed_dim = self.embed_dim
        temperature = self.temperature
        pos_dim = embed_dim // (2 * out_chans)
        omega = torch.arange(pos_dim, dtype=torch.float32, device=x.device) / pos_dim
        omega = 1. / (temperature**omega)

        x = x.permute(1, 0, 2).reshape(out_chans, -1)
        target = target.permute(1, 0, 2).reshape(out_chans, -1)
        fourier_x = torch.einsum('cm,d->cmd', [x, omega])
        fourier_target = torch.einsum('cm,d->cmd', [target, omega])
        fourier_x = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        fourier_target = torch.cat([torch.sin(fourier_target), torch.cos(fourier_target)], dim=1)
        return F.mse_loss(fourier_x, fourier_target, reduction='mean')

class FourierMSELossv2(nn.Module):
    def __init__(self, embed_dim, temperature=0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
    
    def forward(self, x, target):
        # output channels
        out_chans = target.size(-2)
        # embedding dimension and temperature
        embed_dim = self.embed_dim
        temperature = self.temperature
        pos_dim = embed_dim // (2 * out_chans)
        omega = torch.arange(pos_dim, dtype=torch.float32, device=x.device) / pos_dim
        omega = 2. * math.pi / (temperature**omega)

        target = target.permute(0,2,1)
        fourier_target = torch.einsum('bmc,d->bmcd', [target, omega])
        fourier_target = torch.cat([torch.sin(fourier_target), torch.cos(fourier_target)], dim=-1)
        return F.mse_loss(x.flatten(), fourier_target.flatten(), reduction='mean')

class MAE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 encoder, 
                 decoder, 
                 args):
        super().__init__()
        self.args = args
        self.grid_size = grid_size = args.input_size // args.patch_size

        # build positional encoding for encoder and decoder
        with torch.no_grad():
            self.encoder_pos_embed = build_2d_sincos_position_embedding(grid_size, 
                                                                        args.encoder_embed_dim, 
                                                                        num_tokens=1)
            self.decoder_pos_embed = build_2d_sincos_position_embedding(grid_size, 
                                                                        args.decoder_embed_dim, 
                                                                        num_tokens=1)

        # build encoder and decoder
        if args.patchembed.startswith('resnet'):
            import networks
            embed_layer = getattr(networks, args.patchembed)
        else:
            from lib.networks.patch_embed_layers import PatchEmbed2D
            embed_layer = PatchEmbed2D
        self.encoder = encoder(patch_size=args.patch_size,
                               in_chans=args.in_chans,
                               embed_dim=args.encoder_embed_dim,
                               depth=args.encoder_depth,
                               num_heads=args.encoder_num_heads,
                               embed_layer=embed_layer)
        self.decoder = decoder(patch_size=args.patch_size,
                               num_classes=args.fourier_embed_dim * args.patch_size ** 2, # args.in_chans * args.patch_size ** 2,
                               embed_dim=args.decoder_embed_dim,
                               depth=args.decoder_depth,
                               num_heads=args.decoder_num_heads)

        self.encoder_to_decoder = nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))

        # self.patch_norm = nn.LayerNorm(normalized_shape=(args.in_chans * args.patch_size**2,), eps=1e-6, elementwise_affine=False)
        self.patch_norm = nn.LayerNorm(normalized_shape=(args.patch_size**2,), eps=1e-6, elementwise_affine=False)

        # self.criterion = nn.MSELoss()
        print(f"Fourier Embed Dim is {args.fourier_embed_dim}, temperature is {args.fourier_temperature}")
        self.criterion = FourierMSELossv2(embed_dim=args.fourier_embed_dim, temperature=args.fourier_temperature)

        # initialize encoder_to_decoder and mask token
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        # trunc_normal_(self.mask_token, std=.02, a=-.02, b=.02)
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, return_image=False):
        args = self.args
        batch_size = x.size(0)
        in_chans = x.size(1)
        assert in_chans == args.in_chans
        out_chans = in_chans * args.patch_size ** 2
        assert x.size(2) == x.size(3) == args.patch_size * self.grid_size, "Unmatched patch size and grid size"
        x = patchify_image(x, args.patch_size) # [B,gh*gw,C*ph*pw]

        # compute length for selected and masked
        length = self.grid_size ** 2
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        # generate batched shuffle indices
        shuffle_indices = batched_shuffle_indices(batch_size, length, device=x.device)
        unshuffle_indices = shuffle_indices.argsort(dim=1)

        # select and mask the input patches
        shuffled_x = x.gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans))
        sel_x = shuffled_x[:, :sel_length, :]
        msk_x = shuffled_x[:, -msk_length:, :]
        # select and mask the indices
        shuffle_indices = F.pad(shuffle_indices + 1, pad=(1, 0), mode='constant', value=0)
        sel_indices = shuffle_indices[:, :sel_length + 1]
        # msk_indices = shuffle_indices[:, -msk_length:]

        # select the position embedings accordingly
        sel_encoder_pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1).gather(dim=1, index=sel_indices[:, :, None].expand(-1, -1, args.encoder_embed_dim))

        # forward encoder & proj to decoder dimension
        sel_x = self.encoder(sel_x, sel_encoder_pos_embed)
        sel_x = self.encoder_to_decoder(sel_x)

        # combine the selected tokens and mask tokens in the shuffled order
        all_x = torch.cat([sel_x, self.mask_token.expand(batch_size, msk_length, -1)], dim=1)
        # all_x = torch.cat([sel_x[:, :1, :], self.mask_token.expand(batch_size, length, -1)], dim=1)
        # shuffle all the decoder positional encoding
        shuffled_decoder_pos_embed = self.decoder_pos_embed.expand(batch_size, -1, -1).gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, args.decoder_embed_dim))
        # add the shuffled positional embedings to encoder output tokens
        all_x = all_x + shuffled_decoder_pos_embed

        # forward decoder
        all_x = self.decoder(all_x)

        loss = self.criterion(x=all_x[:, -msk_length:, :], 
                              target=msk_x.reshape(-1, in_chans, args.patch_size ** 2).detach())
        # loss = self.criterion(x=all_x[:, -msk_length:, :].reshape(-1, in_chans, args.patch_size ** 2), 
        #                       target=msk_x.reshape(-1, in_chans, args.patch_size ** 2).detach())

        # loss = self.criterion(input=all_x[:, -msk_length, :].reshape(batch_size, -1, in_chans, args.patch_size ** 2), 
        #                       target=self.patch_norm(msk_x.reshape(batch_size, -1, in_chans, args.patch_size ** 2).detach()))
        
        # loss = self.criterion(input=all_x[:, 1:, :].reshape(batch_size, -1, in_chans, args.patch_size ** 2), 
        #                       target=self.patch_norm(shuffled_x.reshape(batch_size, -1, in_chans, args.patch_size ** 2).detach()))

        # loss = self.criterion(input=all_x[:, 1:, :], 
        #                       target=self.patch_norm(shuffled_x.detach()))

        # import pdb
        # pdb.set_trace()

        if return_image:
            # unshuffled all the tokens
            masked_x = torch.cat([shuffled_x[:, :sel_length, :], 0.5 * torch.ones(batch_size, msk_length, out_chans).to(x.device)], dim=1).gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))
            # reshape
            recon = all_x[:, 1:, :].gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))
            # recon = recon * (x.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6) + x.mean(dim=-1, keepdim=True)
            # recon = all_x[:, 1:, :].gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans)).reshape(batch_size, -1, args.in_chans, args.patch_size**2)
            # reshaped_x = x.reshape(batch_size, -1, args.in_chans, args.patch_size**2)
            # recon = recon * (reshaped_x.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6) + reshaped_x.mean(dim=-1, keepdim=True)
            # recon = recon.reshape(batch_size, -1, out_chans)
            # recon = all_x[:, 1:, :].gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))
            return loss, x.detach(), recon.detach(), masked_x.detach()
        else:
            return loss
