import argparse
import os
import json
import warnings
import datetime
import pandas as pd
import torch
from pathlib import Path
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from accelerate import PartialState
from PIL import Image

CHECKPOINT_FILE = "checkpoint.txt"
START_IMAGE_INDEX = -1

def read_checkpoint(folder='.'):
    """read from checkpoint"""
    checkpoint_path = os.path.join(folder, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as file:
            try:
                return int(file.read().strip())
            except ValueError:
                return START_IMAGE_INDEX
    return START_IMAGE_INDEX

def write_checkpoint(image_index, folder='.'):
    """write to checkpoint"""
    checkpoint_path = os.path.join(folder, CHECKPOINT_FILE)
    os.makedirs(folder, exist_ok=True)  # make sure the folder exist
    with open(checkpoint_path, 'w') as file:
        file.write(str(image_index))

def parse_specify(specify_classes):  # convert to dict
    # Format: "cls_1:num_1,cls_2:num2,..."
    clusters = [cn.split(":") for cn in specify_classes.split(",")]
    return {c: int(n) for c, n in clusters}


def generate_images(diffusers,
                    prompts_path, 
                    save_folder,
                    guidance_scale=7.5,
                    image_size=50,
                    ddim_steps=50,
                    num_samples=5,
                    use_cuda_generator=False,
                    specify_classes=None,
                    log_sep=10,
                    show_alpha=False,
                    use_safety_checker=False
                    ):
    os.makedirs(save_folder, exist_ok=True)
    
    last_completed_idx = read_checkpoint(save_folder)
    print(f"starting from index: {last_completed_idx}")
    
    df = pd.read_csv(prompts_path)
    data = [row for _, row in df.iterrows()]
    
    if specify_classes:
        class2num = parse_specify(specify_classes)
    else:
        class2num = {}
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    diffuser = diffusers.to(device)
    
    if use_cuda_generator and "cuda" in device:
        get_generator = lambda seed: torch.Generator(device=device).manual_seed(seed)
    else:
        if use_cuda_generator:
            warnings.warn(
                "Warning: use_cuda_generator will be ignored because CUDA is not used."
            )
        get_generator = lambda seed: torch.manual_seed(seed)
    
    for idx, row in enumerate(data):
        if idx <= last_completed_idx:
            continue
      

        if 'class' in row and getattr(row, 'class') in class2num:
            gen_num_samples = class2num[getattr(row, 'class')]
        else:
            gen_num_samples = num_samples
            
        prompts = str(row.prompt)
        seed = int(row.evaluation_seed)
        case_number = int(row.case_number)
        print(f"id: {case_number}")
        print(prompts)
        images = diffuser(
            prompts=prompts,
            img_size=image_size,
            n_steps=ddim_steps,
            n_imgs=gen_num_samples,
            guidance_scale=guidance_scale,
            end_iteration=None,
            generator=get_generator(seed),
            show_alpha=show_alpha,
            use_safety_checker=use_safety_checker,
        )
        index = idx
        for idx, image_tuple in enumerate(images):
            image = image_tuple[0]
            if isinstance(image, Image.Image):
                image.save(f"{save_folder}/{case_number}_{idx}.png")
            else:
                print("Invalid image object found.")
                
        write_checkpoint(index, save_folder)