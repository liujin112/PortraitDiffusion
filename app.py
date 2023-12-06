import os
import torch
import random
import numpy as np
import gradio as gr
from glob import glob
from datetime import datetime

from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler, LCMScheduler

import torch.nn.functional as F
from PIL import Image,ImageDraw
from utils.masactrl_utils import (AttentionBase,
                                     regiter_attention_editor_diffusers)
from utils.free_lunch_utils import register_upblock2d,register_crossattn_upblock2d,register_free_upblock2d, register_free_crossattn_upblock2d
from utils.style_attn_control import MaskPromptedStyleAttentionControl
from utils.pipeline import MasaCtrlPipeline
from torchvision.utils import save_image
from segment_anything import sam_model_registry, SamPredictor



css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

class GlobalText:
    def __init__(self):
        
        # config dirs
        self.basedir                = os.getcwd()
        self.stable_diffusion_dir   = os.path.join(self.basedir, "models", "StableDiffusion")
        self.personalized_model_dir = './models/Stable-diffusion'
        self.lora_model_dir         = './models/Lora'
        self.savedir                = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample         = os.path.join(self.savedir, "sample")

        self.savedir_mask         = os.path.join(self.savedir, "mask")

        self.stable_diffusion_list   = ["runwayml/stable-diffusion-v1-5",
                                        "latent-consistency/lcm-lora-sdv1-5"]
        self.personalized_model_list = []
        self.lora_model_list = []

        # config models
        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.unet                  = None
        self.pipeline              = None
        self.lora_loaded           = None
        self.lcm_lora_loaded       = False
        self.personal_model_loaded = None
        self.sam_predictor         = None

        self.lora_model_state_dict = {}
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.refresh_stable_diffusion()
        self.refresh_personalized_model()

        

        self.reset_start_code()
    def load_base_pipeline(self, model_path):
        

        print(f'loading {model_path} model')
        scheduler = DDIMScheduler.from_pretrained(model_path,subfolder="scheduler")
        self.pipeline = MasaCtrlPipeline.from_pretrained(model_path,
                                         scheduler=scheduler).to(self.device)
        
    def refresh_stable_diffusion(self):
        
        self.load_base_pipeline(self.stable_diffusion_list[0])
        self.lora_loaded           = None
        self.personal_model_loaded = None
        self.lcm_lora_loaded       = False
        return self.stable_diffusion_list[0]
    
    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(self.personalized_model_dir, "**/*.safetensors"), recursive=True)
        self.personalized_model_list = {os.path.basename(file): file for file in personalized_model_list}
        
        lora_model_list = glob(os.path.join(self.lora_model_dir, "**/*.safetensors"), recursive=True)
        self.lora_model_list = {os.path.basename(file): file for file in lora_model_list}
        
    def update_stable_diffusion(self, stable_diffusion_dropdown):
        
        if stable_diffusion_dropdown == 'latent-consistency/lcm-lora-sdv1-5':
            self.load_lcm_lora()
        else:
            self.load_base_pipeline(stable_diffusion_dropdown)
        self.lora_loaded           = None
        self.personal_model_loaded = None
        return gr.Dropdown()
    
    def update_base_model(self, base_model_dropdown):
        if self.pipeline is None:
            gr.Info(f"Please select a pretrained model path.")
            return None
        else:
            base_model = self.personalized_model_list[base_model_dropdown]
            mid_model = StableDiffusionPipeline.from_single_file(base_model)
            self.pipeline.vae = mid_model.vae
            self.pipeline.unet = mid_model.unet
            self.pipeline.text_encoder = mid_model.text_encoder
            self.pipeline.to(self.device)
            self.personal_model_loaded = base_model_dropdown.split('.')[0]
            print(f'load {base_model_dropdown} model success!')
            
            return gr.Dropdown()

    
    def update_lora_model(self, lora_model_dropdown,lora_alpha_slider):
        
        if self.pipeline is None:
            gr.Info(f"Please select a pretrained model path.")
            return None
        else:
            if lora_model_dropdown == "none":
                self.pipeline.unfuse_lora()
                self.pipeline.unload_lora_weights()
                self.lora_loaded           = None
                print("Restore lora.")
            else:
                
                lora_model_path = self.lora_model_list[lora_model_dropdown]
                self.pipeline.load_lora_weights(lora_model_path)
                self.pipeline.fuse_lora(lora_alpha_slider)
                self.lora_loaded = lora_model_dropdown.split('.')[0]
                print(f'load {lora_model_dropdown} LoRA Model Success!')
        return gr.Dropdown()
    
    def load_lcm_lora(self, lora_alpha_slider=1.0):
        # set scheduler
        self.pipeline = MasaCtrlPipeline.from_pretrained(self.stable_diffusion_list[0]).to(self.device)
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)
        # load LCM-LoRA
        self.pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.pipeline.fuse_lora(lora_alpha_slider)
        self.lcm_lora_loaded = True
        print(f'load LCM-LoRA model success!')
        
    def generate(self, source, style, source_mask, style_mask,  
                       start_step, start_layer, Style_attn_step,
                       Method, Style_Guidance, ddim_steps, scale, seed, de_bug,
                       target_prompt, negative_prompt_textbox,
                       inter_latents,
                       freeu, b1, b2, s1, s2,
                       width_slider,height_slider,
                       ):
        os.makedirs(self.savedir, exist_ok=True)
        os.makedirs(self.savedir_sample, exist_ok=True)
        os.makedirs(self.savedir_mask, exist_ok=True)
        model = self.pipeline

        if seed != -1 and seed != "": torch.manual_seed(int(seed))
        else: torch.seed()
        seed = torch.initial_seed()
        sample_count = len(os.listdir(self.savedir_sample))
        os.makedirs(os.path.join(self.savedir_mask, f"results_{sample_count}"), exist_ok=True)

        # ref_prompt = [source_prompt, target_prompt]
        # prompts = ref_prompt+['']
        ref_prompt = [target_prompt, target_prompt]
        prompts = ref_prompt+[target_prompt]    
        source_image,style_image,source_mask,style_mask = load_mask_images(source,style,source_mask,style_mask,self.device,width_slider,height_slider,out_dir=os.path.join(self.savedir_mask, f"results_{sample_count}"))

        
        # global START_CODE, LATENTS_LIST

        with torch.no_grad():
            #import pdb;pdb.set_trace()

            #prev_source
            if self.start_code is None and self.latents_list is None:
                content_style = torch.cat([style_image, source_image], dim=0)
                editor = AttentionBase()
                regiter_attention_editor_diffusers(model, editor)  
                st_code, latents_list = model.invert(content_style,
                                                        ref_prompt,
                                                        guidance_scale=scale,
                                                        num_inference_steps=ddim_steps,
                                                        return_intermediates=True)
                start_code = torch.cat([st_code, st_code[1:]], dim=0)
                self.start_code = start_code
                self.latents_list = latents_list
            else:
                start_code = self.start_code
                latents_list = self.latents_list
                print('------------------------------------------  Use previous latents ------------------------------------------  ')
            
            #["Without mask", "Only masked region", "Seperate Background Foreground"]
            
            if Method == "Without mask":
                style_mask = None
                source_mask = None
                only_masked_region = False
            elif Method == "Only masked region":
                assert style_mask is not None and source_mask is not None
                only_masked_region = True
            else:
                assert style_mask is not None and source_mask is not None
                only_masked_region = False
            
            controller = MaskPromptedStyleAttentionControl(start_step, start_layer,
                                                        style_attn_step=Style_attn_step,
                                                        style_guidance=Style_Guidance, 
                                                        style_mask=style_mask,
                                                        source_mask=source_mask,
                                                        only_masked_region=only_masked_region, 
                                                        guidance=scale,
                                                        de_bug=de_bug,
                                                        )
            if freeu:
                # model.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
                print(f'++++++++++++++++++ Run with FreeU {b1}_{b2}_{s1}_{s2} ++++++++++++++++')
                if Method != "Without mask":
                    register_free_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s1,source_mask=source_mask)
                    register_free_crossattn_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s1,source_mask=source_mask)
                else:
                    register_free_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s1,source_mask=None)
                    register_free_crossattn_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s1,source_mask=None)
                
            else:
                print(f'++++++++++++++++++ Run without FreeU ++++++++++++++++')
                # model.disable_freeu()
                register_upblock2d(model)
                register_crossattn_upblock2d(model)
            
            regiter_attention_editor_diffusers(model, controller)

            

            # inference the synthesized image
            generate_image= model(prompts,
                                width=width_slider,
                                height=height_slider,
                                latents=start_code,
                                guidance_scale=scale,
                                num_inference_steps=ddim_steps,
                                ref_intermediate_latents=latents_list if inter_latents else None,
                                neg_prompt=negative_prompt_textbox,
                                return_intermediates=False,
                                lcm_lora=self.lcm_lora_loaded,
                                de_bug=de_bug,)
            
            # os.makedirs(os.path.join(output_dir, f"results_{sample_count}"))
            save_file_name = f"results_{sample_count}_step{start_step}_layer{start_layer}SG{Style_Guidance}_style_attn_step{Style_attn_step}.jpg"
            if self.lora_loaded != None:
                save_file_name = f"lora_{self.lora_loaded}_" + save_file_name
            if self.personal_model_loaded != None:
                save_file_name = f"personal_{self.personal_model_loaded}_" + save_file_name
                #f"results_{sample_count}_step{start_step}_layer{start_layer}SG{Style_Guidance}_style_attn_step{Style_attn_step}_lora_{self.lora_loaded}.jpg"
            save_file_path = os.path.join(self.savedir_sample, save_file_name)
            #save_file_name = os.path.join(output_dir, f"results_style_{style_name}", f"{content_name}.jpg")
                    
            save_image(torch.cat([source_image/2 + 0.5, style_image/2 + 0.5, generate_image[2:]], dim=0), save_file_path, nrow=3, padding=0)
                    

            
            # global OUTPUT_RESULT
            # OUTPUT_RESULT = save_file_name

            generate_image = generate_image.cpu().permute(0, 2, 3, 1).numpy()
            #save_gif(latents_list, os.path.join(output_dir, f"results_{sample_count}",'output_latents_list.gif'))
        # import pdb;pdb.set_trace()
            #gif_dir = os.path.join(output_dir, f"results_{sample_count}",'output_latents_list.gif')
            
        return [
            generate_image[0],
            generate_image[1],
            generate_image[2],
            ]
        
    def reset_start_code(self,):
        self.start_code = None
        self.latents_list = None

    def lora_sam_predictor(self, sam_path):
        sam_checkpoint = sam_path
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        self.sam_point = []
        self.sam_point_label = []
        
    def get_points_with_draw(self, image, image_with_points, label, evt: gr.SelectData):

        x, y = evt.index[0], evt.index[1]
        point_radius, point_color = 15, (255, 255, 0) if label == 'Add Mask' else (255, 0, 255)
        self.sam_point.append([x, y])
        self.sam_point_label.append(1 if label == 'Add Mask' else 0)
        
        print(x, y, label == 'Add Mask')
        
        if image_with_points is None:
            draw = ImageDraw.Draw(image)
            draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
            return image
        else:
        
            draw = ImageDraw.Draw(image_with_points)
            draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
            return image_with_points
    def reset_sam_points(self,):
        self.sam_point = []
        self.sam_point_label = []
        print('reset all points')
        return None
    def obtain_mask(self, image,sam_path):
        if self.sam_predictor is None:
            self.lora_sam_predictor(sam_path)
        
        print("+++++++++++++++++++ Obtain Mask by SAM ++++++++++++++++++++++")
        input_point = np.array(self.sam_point)
        input_label = np.array(self.sam_point_label)
        predictor = self.sam_predictor
        image = np.array(image)
        predictor.set_image(image)

        # input_point = np.array([[500, 375]])
        # input_label = np.array([1])

        masks, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label,multimask_output=False)   
        
        # import pdb; pdb.set_trace()
        masks = masks.astype(np.uint8)
        masks = masks * 255
        masks = masks.transpose(1,2,0)
        masks = masks.repeat(3, axis=2)
        return masks

global_text = GlobalText()


def load_mask_images(source,style,source_mask,style_mask,device,width,height,out_dir=None):
    # invert the image into noise map
    if isinstance(source['image'], np.ndarray):
        source_image = torch.from_numpy(source['image']).to(device) / 127.5 - 1.
    else:
        source_image = torch.from_numpy(np.array(source['image'])).to(device) / 127.5 - 1.
    source_image = source_image.unsqueeze(0).permute(0, 3, 1, 2)

    source_image = F.interpolate(source_image, (height,width ))

    if out_dir is not None and source_mask is None:
        
        source['mask'].save(os.path.join(out_dir,'source_mask.jpg'))
    else:
        Image.fromarray(source_mask).save(os.path.join(out_dir,'source_mask.jpg'))
    if out_dir is not None and style_mask is None:
        
        style['mask'].save(os.path.join(out_dir,'style_mask.jpg'))
    else:
        Image.fromarray(style_mask).save(os.path.join(out_dir,'style_mask.jpg'))
    
    source_mask = torch.from_numpy(np.array(source['mask']) if source_mask is None else source_mask).to(device) / 255.
    source_mask = source_mask.unsqueeze(0).permute(0, 3, 1, 2)[:,:1]
    source_mask = F.interpolate(source_mask, (height//8,width//8))

    if isinstance(source['image'], np.ndarray):
        style_image = torch.from_numpy(style['image']).to(device) / 127.5 - 1.
    else:
        style_image = torch.from_numpy(np.array(style['image'])).to(device) / 127.5 - 1.
    style_image = style_image.unsqueeze(0).permute(0, 3, 1, 2)
    style_image = F.interpolate(style_image, (height,width))
    
    style_mask = torch.from_numpy(np.array(style['mask']) if style_mask is None else style_mask ).to(device) / 255.
    style_mask = style_mask.unsqueeze(0).permute(0, 3, 1, 2)[:,:1]
    style_mask = F.interpolate(style_mask, (height//8,width//8))

    
    return source_image,style_image,source_mask,style_mask


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # [Portrait Diffusion: Training-free Face Stylization with Chain-of-Painting](https://arxiv.org/abs/2312.02212)
            Jin Liu, Huaibo Huang, Chao Jin, Ran He* (*Corresponding Author)<br>
            [Arxiv Report](https://arxiv.org/abs/2312.02212) | [Github](https://github.com/liujin112/PortraitDiffusion)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Select a pretrained model.
                """
            )
            with gr.Row():
                stable_diffusion_dropdown = gr.Dropdown(
                    label="Pretrained Model Path",
                    choices=global_text.stable_diffusion_list,
                    interactive=True,
                    allow_custom_value=True
                )
                stable_diffusion_dropdown.change(fn=global_text.update_stable_diffusion, inputs=[stable_diffusion_dropdown], outputs=[stable_diffusion_dropdown])
                
                stable_diffusion_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_stable_diffusion():
                    global_text.refresh_stable_diffusion()
                    
                stable_diffusion_refresh_button.click(fn=update_stable_diffusion, inputs=[], outputs=[])

                base_model_dropdown = gr.Dropdown(
                    label="Select a ckpt model (optional)",
                    choices=sorted(list(global_text.personalized_model_list.keys())),
                    interactive=True,
                    allow_custom_value=True,
                )
                base_model_dropdown.change(fn=global_text.update_base_model, inputs=[base_model_dropdown], outputs=[base_model_dropdown])
                
                lora_model_dropdown = gr.Dropdown(
                    label="Select a LoRA model (optional)",
                    choices=["none"] + sorted(list(global_text.lora_model_list.keys())),
                    value="none",
                    interactive=True,
                    allow_custom_value=True,
                )
                lora_alpha_slider = gr.Slider(label="LoRA alpha", value=0.8, minimum=0, maximum=2, interactive=True)
                lora_model_dropdown.change(fn=global_text.update_lora_model, inputs=[lora_model_dropdown,lora_alpha_slider], outputs=[lora_model_dropdown])
                
                
                
                personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                
                def update_personalized_model():
                    global_text.refresh_personalized_model()
                    return [
                        gr.Dropdown(choices=sorted(list(global_text.personalized_model_list.keys()))),
                        gr.Dropdown(choices=["none"] + sorted(list(global_text.lora_model_list.keys())))
                    ]
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[base_model_dropdown, lora_model_dropdown])


        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for PortraitDiff.
                """
            )
            with gr.Tab("Configs"):

                with gr.Row():
                    source_image = gr.Image(label="Source Image",  elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGB", height=512)
                    style_image = gr.Image(label="Style Image", elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGB", height=512)
                with gr.Row():
                    prompt_textbox = gr.Textbox(label="Prompt", value='head', lines=1)
                    negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=1)
                    # output_dir = gr.Textbox(label="output_dir", value='./results/')

                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        width_slider     = gr.Slider(label="Width", value=512, minimum=256, maximum=1024, step=64)
                        height_slider    = gr.Slider(label="Height", value=512, minimum=256, maximum=1024, step=64)
                        Method = gr.Dropdown(
                                    ["Without mask", "Only masked region", "Seperate Background Foreground"],
                                    value="Without mask",
                                    label="Mask", info="Select how to use masks")
                        with gr.Tab('Base Configs'):
                            with gr.Row():
                                # sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                                ddim_steps = gr.Slider(label="DDIM Steps", value=50, minimum=0, maximum=100, step=1)
                                
                                Style_attn_step = gr.Slider(label="Step of Style Attention Control",
                                                        minimum=0,
                                                        maximum=50,
                                                        value=35,
                                                        step=1)
                                start_step = gr.Slider(label="Step of Attention Control",
                                                        minimum=0,
                                                        maximum=150,
                                                        value=0,
                                                        step=1)
                                start_layer = gr.Slider(label="Layer of Style Attention Control",
                                                        minimum=0,
                                                        maximum=16,
                                                        value=10,
                                                        step=1)
                                Style_Guidance = gr.Slider(label="Style Guidance Scale",
                                                minimum=0,
                                                maximum=4,
                                                value=1.2,
                                                step=0.05)
                                cfg_scale_slider = gr.Slider(label="CFG Scale",        value=0, minimum=0,   maximum=20)

                                
                        with gr.Tab('FreeU'):
                            with gr.Row():
                                freeu = gr.Checkbox(label="Free Upblock", value=False)
                                de_bug = gr.Checkbox(value=False,label='DeBug')
                                inter_latents = gr.Checkbox(value=True,label='Use intermediate latents')
                            with gr.Row():
                                b1 = gr.Slider(label='b1:',
                                                        minimum=-1,
                                                        maximum=2,
                                                        step=0.01,
                                                        value=1.3)
                                b2 = gr.Slider(label='b2:',
                                                        minimum=-1,
                                                        maximum=2,
                                                        step=0.01,
                                                        value=1.5)
                            with gr.Row():
                                s1 = gr.Slider(label='s1: ',
                                                        minimum=0,
                                                        maximum=2,
                                                        step=0.1,
                                                        value=1.0)
                                s2 = gr.Slider(label='s2:',
                                                        minimum=0,
                                                        maximum=2,
                                                        step=0.1,
                                                        value=1.0)
                        with gr.Row():
                            seed_textbox = gr.Textbox(label="Seed", value=-1)
                            seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                            seed_button.click(fn=lambda: random.randint(1, 1e8), inputs=[], outputs=[seed_textbox])

                    with gr.Column():
                        generate_button = gr.Button(value="Generate", variant='primary')

                        generate_image = gr.Image(label="Image with PortraitDiff", interactive=False, type='numpy', height=512,)

                        with gr.Row():
                            recons_content = gr.Image(label="reconstructed content", type="pil", image_mode="RGB", height=256)
                            recons_style = gr.Image(label="reconstructed style", type="pil", image_mode="RGB", height=256)

            with gr.Tab("SAM"):
                with gr.Column():
                    with gr.Row():
                        add_or_remove = gr.Radio(["Add Mask", "Remove Area"], value="Add Mask", label="Point_label (foreground/background)")
                        sam_path = gr.Textbox(label="Sam Model path", value='')
                        load_sam_btn = gr.Button(value="Lora SAM form path")
                    with gr.Row():
                        
                        send_source_btn = gr.Button(value="Send Source Image from PD Tab")
                        sam_source_btn = gr.Button(value="Segment Source")
                        
                        send_style_btn = gr.Button(value="Send Style Image from PD Tab")
                        sam_style_btn = gr.Button(value="Segment Style")
                    with gr.Row():
                        source_image_sam = gr.Image(label="Source Image SAM",  elem_id="SourceimgSAM", source="upload", interactive=True, type="pil", image_mode="RGB", height=512)
                        style_image_sam = gr.Image(label="Style Image SAM", elem_id="StyleimgSAM", source="upload", interactive=True, type="pil", image_mode="RGB", height=512)

                    with gr.Row():
                        source_image_with_points = gr.Image(label="source Image with points", elem_id="style_image_with_points", type="pil", image_mode="RGB", height=256)
                        source_mask = gr.Image(label="Source Mask",  elem_id="img2maskimg", source="upload", interactive=True, type="numpy", image_mode="RGB", height=256)

                        style_image_with_points = gr.Image(label="Style Image with points", elem_id="style_image_with_points", type="pil", image_mode="RGB", height=256)
                        style_mask = gr.Image(label="Style Mask", elem_id="img2maskimg", source="upload", interactive=True, type="numpy", image_mode="RGB", height=256)
                load_sam_btn.click(global_text.lora_sam_predictor,inputs=[sam_path],outputs=[])
                source_image_sam.select(global_text.get_points_with_draw, [source_image_sam, source_image_with_points, add_or_remove], source_image_with_points)
                style_image_sam.select(global_text.get_points_with_draw, [style_image_sam, style_image_with_points, add_or_remove], style_image_with_points)
                send_source_btn.click(lambda x: (x['image'], None), inputs=[source_image], outputs=[source_image_sam, source_image_with_points])
                send_style_btn.click(lambda x: (x['image'], None), inputs=[style_image], outputs=[style_image_sam, style_image_with_points])
                
                style_image_sam.change(global_text.reset_sam_points, inputs=[], outputs=[style_image_with_points])
                source_image_sam.change(global_text.reset_sam_points, inputs=[], outputs=[source_image_with_points])
                
                
                sam_source_btn.click(global_text.obtain_mask,[source_image_sam, sam_path],[source_mask])
                sam_style_btn.click(global_text.obtain_mask,[style_image_sam, sam_path],[style_mask])

        gr.Examples(
            [[os.path.join(os.path.dirname(__file__), "images/content/1.jpg"),
              os.path.join(os.path.dirname(__file__), "images/style/1.jpg")],
            
            ],
            [source_image, style_image]
        )
        inputs = [
            source_image, style_image, source_mask, style_mask,
            start_step, start_layer, Style_attn_step, 
            Method, Style_Guidance,ddim_steps, cfg_scale_slider, seed_textbox, de_bug,  
            prompt_textbox, negative_prompt_textbox, inter_latents,
            freeu, b1, b2, s1, s2,
            width_slider,height_slider
        ]

        generate_button.click(
            fn=global_text.generate,
            inputs=inputs,
            outputs=[recons_style,recons_content,generate_image]
        )
        source_image.upload(global_text.reset_start_code, inputs=[], outputs=[])
        style_image.upload(global_text.reset_start_code, inputs=[], outputs=[])
        ddim_steps.change(fn=global_text.reset_start_code, inputs=[], outputs=[])
    return demo

if __name__ == "__main__":
    demo = ui()
    demo.launch(server_name="172.18.32.44")