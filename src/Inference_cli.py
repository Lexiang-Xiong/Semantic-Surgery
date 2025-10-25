# Inference_cli.py
import argparse
import json
import importlib
import sys
from pathlib import Path
from utils import StableDiffuser
from SS_inference import generate_images

def load_detector(detector_config):
    if not detector_config:
        return None
    
    detector_type = detector_config.get('type')
    if detector_type == "NudeDetection":
        from detection_nude import NudeDetection
        return NudeDetection(**detector_config.get('params', {}))
    elif detector_type == "ConcurrentObjectDetection":
        from detection_aod import ConcurrentObjectDetection
        return ConcurrentObjectDetection(**detector_config.get('params', {}))
    elif detector_type == "custom":
        module_path = detector_config['module']
        class_name = detector_config['class']
        
        if 'path' in detector_config:
            sys.path.append(str(Path(detector_config['path']).resolve()))
            
        module = importlib.import_module(module_path)
        detector_class = getattr(module, class_name)
        return detector_class(**detector_config.get('args', {}))
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

def setup_diffuser(config):
    diffuser_config = config['diffuser']
    
    # init parameters
    diffuser = StableDiffuser(
        scheduler=diffuser_config.get('scheduler', 'DDIM'),
        concepts_to_erase=diffuser_config.get('concepts_to_erase', []),
        neutral_concept=diffuser_config.get('neutral_concept', ''),
        params=diffuser_config.get('params', {})
    ).to(diffuser_config.get('device', 'cuda:0'))
    
    # set detector
    if 'detector' in config:
        detector = load_detector(config['detector'])
        if detector:
            diffuser.detect_method = detector
    
    return diffuser

def main():
    parser = argparse.ArgumentParser(description='Stable Diffusion Batch Inference')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    args = parser.parse_args()

    # read config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # initialization
    diffuser = setup_diffuser(config)
    
    # get generation parameters
    gen_params = config['generation']
    
    # run
    generate_images(
        diffusers=diffuser,
        prompts_path=gen_params['prompts_path'],
        save_folder=gen_params['save_folder'],
        guidance_scale=gen_params.get('guidance_scale', 7.5),
        image_size=gen_params.get('image_size', 512),
        ddim_steps=gen_params.get('ddim_steps', 50),
        num_samples=gen_params.get('num_samples', 5),
        use_cuda_generator=gen_params.get('use_cuda_generator', False),
        specify_classes=gen_params.get('specify_classes'),
        log_sep=gen_params.get('log_interval', 10),
        show_alpha=gen_params.get('show_alpha', False),
        use_safety_checker=gen_params.get('use_safety_checker', False)
    )

if __name__ == "__main__":
    main()