# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

"""
Modified from the official implementation of DeiT.
https://github.com/facebookresearch/deit/blob/main/models.py
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from ..build import MODEL
from pycls.core.config import cfg
from .base import BaseTransformerModel
from .common import PatchEmbedding, TransformerLayer, layernorm


@MODEL.register()
class DeiT(BaseTransformerModel):

    def __init__(self):
        super(DeiT, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.patch_embed = PatchEmbedding(img_size=self.img_size, patch_size=self.patch_size, in_channels=self.in_channels, out_channels=self.hidden_dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_tokens = 1 + cfg.DISTILLATION.ENABLE_LOGIT
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.hidden_dim))
        self.pe_dropout = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([TransformerLayer(
            in_channels=self.hidden_dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=dpr[i]) for i in range(self.depth)])
        self.initialize_hooks(self.layers)
        self.norm = layernorm(self.hidden_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(self.hidden_dim, self.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            TransformerLayer(in_channels = self.decoder_embed_dim, num_heads = self.decoder_num_heads, mlp_ratio = self.mlp_ratio, qkv_bias=True)
            for i in range(self.decoder_depth)])
        self.decoder_norm = layernorm(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.in_chans, bias=True) # decoder to patch
 #################################################
        self.apply(self._init_weights)

        self.head = nn.Linear(self.hidden_dim, self.num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.distill_logits = None

        self.distill_token = None
        self.distill_head = None
        if cfg.DISTILLATION.ENABLE_LOGIT:
            self.distill_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.distill_head = nn.Linear(self.hidden_dim, self.num_classes)
            nn.init.zeros_(self.distill_head.weight)
            nn.init.constant_(self.distill_head.bias, 0)
            trunc_normal_(self.distill_token, std=.02)

    def _feature_hook(self, module, inputs, outputs):
        feat_size = int(self.num_patches ** 0.5)
        x = outputs[:, self.num_tokens:].view(outputs.size(0), feat_size, feat_size, self.hidden_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        self.features.append(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_vanilla(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

    
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.layers:
            x = blk(x)
        x = self.norm(x)

        return x
        

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x):
        MASK_RATIO = 0.75
        mask_ratio = MASK_RATIO
        if self.training:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            pred_decoder = self.forward_decoder(latent, ids_restore)
            loss_twobranch = self.forward_loss(imgs, pred_decoder , mask)
        x = self.patch_embed(x)
        if self.num_tokens == 1:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        else:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), self.distill_token.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pe_dropout(x + self.pos_embed)
        
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if self.num_tokens == 1:
            if self.training:
                return logits , loss_twobranch
            else:
                return logits

        self.distill_logits = None
        self.distill_logits = self.distill_head(x[:, 1])

        if self.training:
            return logits , loss_twobranch
        else:
            return (logits + self.distill_logits) / 2
