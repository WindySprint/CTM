# -*- coding: utf-8 -*-
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

logger = logging.getLogger(__name__)

class W_SW_Attention(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(W_SW_Attention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type

        self.embedding_layer1 = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.embedding_layer2 = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        self.relative_position_params1 = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))
        trunc_normal_(self.relative_position_params1, std=.02)
        self.relative_position_params1 = torch.nn.Parameter(
            self.relative_position_params1.view(2 * window_size - 1, 2 * window_size - 1,
                                               self.n_heads).transpose(1, 2).transpose(0, 1))

        self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        self.linear2 = nn.Linear(self.input_dim, self.output_dim)

        self.relative_position_params2 = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))
        trunc_normal_(self.relative_position_params2, std=.02)
        self.relative_position_params2 = torch.nn.Parameter(
            self.relative_position_params2.view(2 * window_size - 1, 2 * window_size - 1,
                                                self.n_heads).transpose(1, 2).transpose(0, 1))


    def generate_mask(self, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(w, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params1.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, emb1,emb2):
        #input: [2, 128, 128, 256] [2, 128, 128, 128] [2, 128, 128, 384]
        if self.type != 'W':
            emb1 = torch.roll(emb1, shifts=(-(self.window_size // 2)), dims=1)
            emb2 = torch.roll(emb2, shifts=(-(self.window_size // 2)), dims=1)
        emb1 = rearrange(emb1, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h1_windows = emb1.size(1)
        w1_windows = emb1.size(2)
        # sqaure validation
        assert h1_windows == w1_windows

        emb2 = rearrange(emb2, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h2_windows = emb2.size(1)
        w2_windows = emb2.size(2)
        # sqaure validation
        assert h2_windows == w2_windows

        emb1 = rearrange(emb1, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        emb2 = rearrange(emb2, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)

        qkv1 = self.embedding_layer1(emb1)
        qkv2 = self.embedding_layer2(emb2)
        q1, k1, v1 = rearrange(qkv1, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        q2, k2, v2 = rearrange(qkv2, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)

        sim1 = torch.einsum('hbwpc,hbwqc->hbwpq', q1, k2) * self.scale
        sim2 = torch.einsum('hbwpc,hbwqc->hbwpq', q2, k1) * self.scale

        # Adding learnable relative embedding
        sim1 = sim1 + rearrange(self.relative_embedding1(), 'h p q -> h 1 1 p q')
        sim2 = sim2 + rearrange(self.relative_embedding2(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask1 = self.generate_mask(h1_windows, self.window_size, shift=self.window_size // 2)
            sim1 = sim1.masked_fill_(attn_mask1, float("-inf"))
            attn_mask2 = self.generate_mask(h2_windows, self.window_size, shift=self.window_size // 2)
            sim2 = sim2.masked_fill_(attn_mask2, float("-inf"))

        probs1 = nn.functional.softmax(sim1, dim=-1)
        probs2 = nn.functional.softmax(sim2, dim=-1)

        # print(attention_probs4.size())

        output1 = torch.einsum('hbwij,hbwjc->hbwic', probs1, v2)
        output2 = torch.einsum('hbwij,hbwjc->hbwic', probs2, v1)
        output1 = rearrange(output1, 'h b w p c -> b w p (h c)')
        output2 = rearrange(output2, 'h b w p c -> b w p (h c)')

        output1 = self.linear1(output1)
        output2 = self.linear2(output2)
        output1 = rearrange(output1, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h1_windows, p1=self.window_size)
        output2 = rearrange(output2, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h2_windows, p1=self.window_size)

        if self.type != 'W':
            output1 = torch.roll(output1, shifts=(self.window_size // 2), dims=1)
            output2 = torch.roll(output2, shifts=(self.window_size // 2), dims=1)

        return output1, output2

    def relative_embedding1(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params1[:, relation[:, :, 0].long(), relation[:, :, 1].long()]

    def relative_embedding2(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params2[:, relation[:, :, 0].long(), relation[:, :, 1].long()]

class Block_ViT(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(Block_ViT, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type

        # print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1_1 = nn.LayerNorm(input_dim)
        self.ln1_2 = nn.LayerNorm(input_dim)
        self.w_sw_attn = W_SW_Attention(input_dim, input_dim, head_dim, window_size, self.type)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2_1 = nn.LayerNorm(input_dim)
        self.ln2_2 = nn.LayerNorm(input_dim)
        self.mlp_1 = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, emb1,emb2):
        org1 = emb1
        org2 = emb2
        x1 = self.ln1_1(emb1)
        x2 = self.ln1_2(emb2)

        x1,x2 = self.w_sw_attn(x1,x2)
        x1 = org1 + x1
        x2 = org2 + x2

        x1 = x1+self.mlp_1(self.ln2_1(x1))
        x2 = x2+self.mlp_2(self.ln2_2(x2))

        return x1, x2


class Swin_Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, num_layers):
        super(Swin_Block, self).__init__()
        self.layer = nn.ModuleList()

        for i in range(num_layers):
            type = 'W' if not i % 2 else 'SW'
            layer = Block_ViT(input_dim, output_dim, head_dim, window_size, type)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2):
        for layer_block in self.layer:
            emb1,emb2 = layer_block(emb1,emb2)
        return emb1,emb2


class SwinChannel(nn.Module):
    def __init__(self, base_channel):
        super().__init__()
        self.base_channel = base_channel
        self.down_stage1 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2),
                             nn.LayerNorm(4 * base_channel), nn.Linear(4 * base_channel, 2 * base_channel, bias=False)]
        self.down_stage2 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2),
                             nn.LayerNorm(8 * base_channel), nn.Linear(8 * base_channel, 4 * base_channel, bias=False)]

        self.down_stage1 = nn.Sequential(*self.down_stage1)
        self.down_stage2 = nn.Sequential(*self.down_stage2)

        self.sb1 = Swin_Block(2 * base_channel, 2 * base_channel, 16, 8, 2)
        self.sb2 = Swin_Block(4 * base_channel, 4 * base_channel, 16, 8, 2)

        self.up_stage1 = [Rearrange('b h w (neiw neih c) -> b (h neih) (w neiw) c', neih=2, neiw=2),
                             nn.LayerNorm(base_channel//2), nn.Linear(base_channel//2, base_channel, bias=False)]
        self.up_stage2 = [Rearrange('b h w (neiw neih c) -> b (h neih) (w neiw) c', neih=2, neiw=2),
                              nn.LayerNorm(base_channel), nn.Linear(base_channel, 2 * base_channel, bias=False)]

        self.up_stage1 = nn.Sequential(*self.up_stage1)
        self.up_stage2 = nn.Sequential(*self.up_stage2)


    def forward(self, en1, en2, en3):
        # [32*256*256] [64*128*128] [128*64*64]
        # [b c h w ]->[b h w c]
        emb1 = Rearrange('b c h w -> b h w c')(en1)
        emb2 = Rearrange('b c h w -> b h w c')(en2)
        emb3 = Rearrange('b c h w -> b h w c')(en3)

        # [2, 256, 256, 32]->[2, 128, 128, 64]
        emb1 = self.down_stage1(emb1)
        emb1, emb2 = self.sb1(emb1, emb2)

        # [2, 128, 128, 64]->[2, 64, 64, 128]
        emb2 = self.down_stage2(emb2)
        emb2, emb3 = self.sb2(emb2, emb3)

        # [2, 128, 128, 64]->[2, 256, 256, 32]
        emb1 = self.up_stage1(emb1)
        # [2, 64, 64, 128]->[2, 128, 128, 64]
        emb2 = self.up_stage2(emb2)

        x1 = Rearrange('b h w c -> b c h w')(emb1)
        x2 = Rearrange('b h w c -> b c h w')(emb2)
        x3 = Rearrange('b h w c -> b c h w')(emb3)

        x1 = x1 + en1
        x2 = x2 + en2
        x3 = x3 + en3

        return x1, x2, x3

