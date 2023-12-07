import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
from diffusers import DDIMScheduler,LCMScheduler
from torchvision.utils import save_image
from torchvision.io import read_image
from PIL import Image 
from utils.pipeline import MasaCtrlPipeline
from utils.masactrl_utils import AttentionBase, regiter_attention_editor_diffusers
from utils.style_attn_control import MaskPromptedStyleAttentionControl


def load_image(image_path, res, device, gray=False):
    image = Image.open(image_path).convert('RGB') if not gray else Image.open(image_path).convert('L')
    image = torch.tensor(np.array(image)).float()
    if gray:
        image = image.unsqueeze(-1).repeat(1,1,3)


    image = image.permute(2, 0, 1)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (res, res))
    image = image.to(device)
    return image

def load_mask(image_path, res, device):
    if image_path != '':
        image = Image.open(image_path).convert('RGB')
        image = torch.tensor(np.array(image)).float()
        image = image.permute(2, 0, 1)
        image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
        image = F.interpolate(image, (res, res))
        image = image.to(device)
        image = image[:, :1, :, :]
    else:
        return None
    return image

def main():
    args = argparse.ArgumentParser()

    args.add_argument("--step", type=int, default=0)
    args.add_argument("--layer", type=int, default=10)
    args.add_argument("--res", type=int, default=512)
    args.add_argument("--style_guidance", type=float, default=1.5)
    args.add_argument("--content", type=str, default=None)
    args.add_argument("--style", type=str, default=None)
    args.add_argument("--content_mask", type=str, default='')
    args.add_argument("--style_mask", type=str, default='')
    args.add_argument("--output", type=str, default='./results/')
    args.add_argument("--only_mask_region", action="store_true")
    args.add_argument("--model_path", type=str, default='runwayml/stable-diffusion-v1-5')
    args.add_argument("--SAC_step", type=int, default=35)
    args.add_argument("--num_inference_steps", type=int, default=50)
    args.add_argument("--LCM_lora", action="store_true")

    args = args.parse_args()
    STEP = args.step
    LAYPER = args.layer
    only_mask_region = args.only_mask_region
    out_dir = args.output
    style_guidance = args.style_guidance
    num_inference_steps = args.num_inference_steps
    SAC_step = args.SAC_step
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    Guidance_scale = 0.0

    
    model_path = args.model_path
    model = MasaCtrlPipeline.from_pretrained(model_path).to(device)
    

    if args.LCM_lora:
        model.scheduler = LCMScheduler.from_config(model.scheduler.config)
        # load LCM-LoRA
        model.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    else:
        model.scheduler = DDIMScheduler.from_config(model.scheduler.config)

    source_image = load_image(args.content, args.res, device)
    style_image = load_image(args.style, args.res, device)

    style_mask = load_mask(args.style_mask, res=64, device=device)
    source_mask = load_mask(args.content_mask, res=args.res, device=device)

    with torch.no_grad():
        
            style_content = torch.cat([style_image, source_image], dim=0)
            source_prompt = ['head', 'head']

            prompts = source_prompt + ['head']

            editor = AttentionBase()
            regiter_attention_editor_diffusers(model, editor)   
            st_code, latents_list = model.invert(style_content,
                                                source_prompt,
                                                guidance_scale=Guidance_scale,
                                                num_inference_steps=num_inference_steps,
                                                return_intermediates=True)
                                                    
            start_code = torch.cat([st_code, st_code[1:]], dim=0)
            assert start_code.shape[0] == 3
            
            editor = MaskPromptedStyleAttentionControl(STEP, LAYPER, 
                                                     style_attn_step=SAC_step,
                                                     style_guidance=style_guidance, 
                                                     style_mask=style_mask,
                                                     source_mask=source_mask,
                                                     only_masked_region=only_mask_region,
                                                     guidance=Guidance_scale,
                                                    )
            
            regiter_attention_editor_diffusers(model, editor)

            image_masactrl = model(prompts,
                                latents=start_code,
                                guidance_scale=Guidance_scale,
                                ref_intermediate_latents=latents_list,
                                num_inference_steps=num_inference_steps
                                )

            os.makedirs(out_dir, exist_ok=True)
            save_image(image_masactrl, os.path.join(out_dir, f"sample{len(os.listdir(out_dir))}_SACstep{SAC_step}_SG{style_guidance}_Tstep{num_inference_steps}.png"), nrow=3)
            print("Syntheiszed images are saved in", out_dir)

if __name__ == "__main__":
    main()