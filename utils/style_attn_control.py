# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from re import U

import numpy as np

from einops import rearrange

from .masactrl_utils import AttentionBase

from torchvision.utils import save_image

import sys

import torch
import torch.nn.functional as F
from torch import nn
import torch.fft as fft

from einops import rearrange, repeat
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
# from masactrl.masactrl import MutualSelfAttentionControl

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None



class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class MaskPromptedStyleAttentionControl(AttentionBase):
    def __init__(self, start_step=4, start_layer=10, style_attn_step=35, layer_idx=None, step_idx=None, total_steps=50, style_guidance=0.1, 
                 only_masked_region=False, guidance=0.0, 
                 style_mask=None, source_mask=None, de_bug=False):
        """
        MaskPromptedSAC
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """

        super().__init__()
        self.total_steps = total_steps
        self.total_layers = 16
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("using MaskPromptStyleAttentionControl")
        print("MaskedSAC at denoising steps: ", self.step_idx)
        print("MaskedSAC at U-Net layers: ", self.layer_idx)
        
        self.de_bug = de_bug
        self.style_guidance = style_guidance
        self.only_masked_region = only_masked_region
        self.style_attn_step = style_attn_step
        self.self_attns = []
        self.cross_attns = []
        self.guidance = guidance
        self.style_mask = style_mask
        self.source_mask = source_mask


    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, q_mask,k_mask, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        
        if q_mask is not None:
            sim = sim.masked_fill(q_mask.unsqueeze(0)==0, -torch.finfo(sim.dtype).max)
            
        if k_mask is not None:
            sim = sim.masked_fill(k_mask.permute(1,0).unsqueeze(0)==0, -torch.finfo(sim.dtype).max)
        
        attn = sim.softmax(-1) if attn is None else attn

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out
    
    def attn_batch_fg_bg(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, q_mask,k_mask, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if q_mask is not None:
            sim_fg = sim.masked_fill(q_mask.unsqueeze(0)==0, -torch.finfo(sim.dtype).max)
            sim_bg = sim.masked_fill(q_mask.unsqueeze(0)==1, -torch.finfo(sim.dtype).max)
        if k_mask is not None:
            sim_fg = sim.masked_fill(k_mask.permute(1,0).unsqueeze(0)==0, -torch.finfo(sim.dtype).max)
            sim_bg = sim.masked_fill(k_mask.permute(1,0).unsqueeze(0)==1, -torch.finfo(sim.dtype).max)
        sim = torch.cat([sim_fg, sim_bg])
        attn = sim.softmax(-1)

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out
   
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):

        """
        Attention forward function
        """
        
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        
        if self.style_mask is not None and self.source_mask is not None:
            #mask = self.aggregate_cross_attn_map(idx=self.cur_token_idx)  # (4, H, W)
            heigh, width = self.style_mask.shape[-2:]
            mask_style = self.style_mask# (H, W)
            mask_source = self.source_mask# (H, W)
            scale = int(np.sqrt(heigh * width / q.shape[1]))
            # res = int(np.sqrt(q.shape[1]))
            spatial_mask_source = F.interpolate(mask_source, (heigh//scale, width//scale)).reshape(-1, 1)
            spatial_mask_style = F.interpolate(mask_style, (heigh//scale, width//scale)).reshape(-1, 1)
            
        else:
            spatial_mask_source=None
            spatial_mask_style=None

        if spatial_mask_style is None or spatial_mask_source is None:
            
            out_s,out_c,out_t = self.style_attn_ctrl(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, spatial_mask_source,spatial_mask_style,**kwargs)
        
        else:
            if self.only_masked_region:
                out_s,out_c,out_t = self.mask_prompted_style_attn_ctrl(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, spatial_mask_source,spatial_mask_style,**kwargs)
            else:
                out_s,out_c,out_t = self.separate_mask_prompted_style_attn_ctrl(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, spatial_mask_source,spatial_mask_style,**kwargs)

        out = torch.cat([out_s,out_c,out_t],dim=0)  
        return out
    

    def style_attn_ctrl(self,q,k,v,sim,attn,is_cross,place_in_unet,num_heads,spatial_mask_source,spatial_mask_style,**kwargs):
        if self.de_bug:
            import pdb; pdb.set_trace()
        
        qs, qc, qt = q.chunk(3)

        out_s = self.attn_batch(qs, k[:num_heads], v[:num_heads], sim[:num_heads], attn[:num_heads], is_cross, place_in_unet, num_heads, q_mask=None,k_mask=None,**kwargs)
        out_c = self.attn_batch(qc, k[:num_heads], v[:num_heads], sim[:num_heads], None, is_cross, place_in_unet, num_heads, q_mask=None,k_mask=None,**kwargs)

        if self.cur_step < self.style_attn_step:
            out_t = self.attn_batch(qc, k[:num_heads], v[:num_heads], sim[:num_heads], None, is_cross, place_in_unet, num_heads, q_mask=None,k_mask=None,**kwargs)
        else:
            out_t = self.attn_batch(qt, k[:num_heads], v[:num_heads], sim[:num_heads], None, is_cross, place_in_unet, num_heads, q_mask=None,k_mask=None,**kwargs)
            if self.style_guidance>=0:
                out_t = out_c + (out_t - out_c) * self.style_guidance
        return out_s,out_c,out_t

    def mask_prompted_style_attn_ctrl(self,q,k,v,sim,attn,is_cross,place_in_unet,num_heads,spatial_mask_source,spatial_mask_style,**kwargs):
        qs, qc, qt = q.chunk(3)
        
        out_s = self.attn_batch(qs, k[:num_heads], v[:num_heads], sim[:num_heads], attn[:num_heads], is_cross, place_in_unet, num_heads, q_mask=None,k_mask=None,**kwargs)
        out_c = self.attn_batch(qc, k[num_heads: 2*num_heads], v[num_heads:2*num_heads], sim[num_heads: 2*num_heads], attn[num_heads: 2*num_heads], is_cross, place_in_unet, num_heads, q_mask=None,k_mask=None, **kwargs)
        out_c_new = self.attn_batch(qc, k[num_heads: 2*num_heads], v[num_heads:2*num_heads], sim[num_heads: 2*num_heads], None, is_cross, place_in_unet, num_heads, q_mask=None,k_mask=None, **kwargs)
        
        if self.de_bug:
            import pdb; pdb.set_trace()

        if self.cur_step < self.style_attn_step:
            out_t = out_c #self.attn_batch(qc, k[:num_heads], v[:num_heads], sim[:num_heads], attn, is_cross, place_in_unet, num_heads, q_mask=spatial_mask_source,k_mask=spatial_mask_style,**kwargs)
        else:
            out_t_fg = self.attn_batch(qt, k[:num_heads], v[:num_heads], sim[:num_heads], None, is_cross, place_in_unet, num_heads, q_mask=spatial_mask_source,k_mask=spatial_mask_style,**kwargs)
            out_c_fg = self.attn_batch(qc, k[:num_heads], v[:num_heads], sim[:num_heads], None, is_cross, place_in_unet, num_heads, q_mask=spatial_mask_source,k_mask=spatial_mask_style,**kwargs)
            if self.style_guidance>=0:
                out_t = out_c_fg + (out_t_fg - out_c_fg) * self.style_guidance 
            
            out_t = out_t * spatial_mask_source + out_c * (1 - spatial_mask_source)

        if self.de_bug:
            import pdb; pdb.set_trace()
        
        # print(torch.sum(out_t* (1 - spatial_mask_source) - out_c * (1 - spatial_mask_source)))
        return out_s,out_c,out_t

    def separate_mask_prompted_style_attn_ctrl(self,q,k,v,sim,attn,is_cross,place_in_unet,num_heads,spatial_mask_source,spatial_mask_style,**kwargs):
        
        if self.de_bug:
            import pdb; pdb.set_trace()
        # To prevent query confusion, render fg and bg according to mask.
        qs, qc, qt = q.chunk(3)
        out_s = self.attn_batch(qs, k[:num_heads], v[:num_heads], sim[:num_heads], attn[:num_heads], is_cross, place_in_unet, num_heads, q_mask=None,k_mask=None,**kwargs)
        if self.cur_step < self.style_attn_step: 
            
            out_c = self.attn_batch_fg_bg(qc, k[:num_heads], v[:num_heads], sim[:num_heads], attn, is_cross, place_in_unet, num_heads, q_mask=spatial_mask_source,k_mask=spatial_mask_style,**kwargs)
            out_c_fg,out_c_bg = out_c.chunk(2)
            out_t = out_c_fg * spatial_mask_source + out_c_bg * (1 - spatial_mask_source)

        else:
            out_t = self.attn_batch_fg_bg(qt, k[:num_heads], v[:num_heads], sim[:num_heads], attn, is_cross, place_in_unet, num_heads, q_mask=spatial_mask_source,k_mask=spatial_mask_style,**kwargs)
            out_c = self.attn_batch_fg_bg(qc, k[:num_heads], v[:num_heads], sim[:num_heads], attn, is_cross, place_in_unet, num_heads, q_mask=spatial_mask_source,k_mask=spatial_mask_style,**kwargs)
            out_t_fg,out_t_bg = out_t.chunk(2)
            out_c_fg,out_c_bg = out_c.chunk(2)
            if self.style_guidance>=0:
                out_t_fg = out_c_fg + (out_t_fg - out_c_fg) * self.style_guidance 
                out_t_bg = out_c_bg + (out_t_bg - out_c_bg) * self.style_guidance 
            out_t = out_t_fg * spatial_mask_source + out_t_bg * (1 - spatial_mask_source)
        
        return out_s,out_t,out_t

    
