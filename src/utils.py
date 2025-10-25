from PIL import Image
from matplotlib import pyplot as plt
import textwrap
import argparse
import torch
import copy
import os
import re
from pathlib import Path
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torch.nn.functional as F

def compute_similarity(embedding1, embedding2):
    # Ensure that the shapes of the two embeddings are consistent.
    assert embedding1.shape == embedding2.shape

    # Calculate the cosine similarity for each position and take the average.
    cos_sim = F.cosine_similarity(embedding1, embedding2, dim=-1).mean(dim=1)
    return cos_sim


def sigmoid_kernel(similarity, gamma, bias):
    # Use a Gaussian kernel to transform similarity into the [0, 1] range
    alpha = torch.sigmoid((similarity + bias) / gamma)
    return alpha


def to_gif(images, path):

    images[0].save(
        path, save_all=True, append_images=images[1:], loop=0, duration=len(images) * 20
    )


def get_selected_alpha(alphas_raw, th):
    # Step 1 and 2: Traverse each tensor, find the maximum value, and collect it into a list.
    alphas = [torch.max(alpha).item() for alpha in alphas_raw]
    
    # Step 3: Filter out values below the threshold and record values that are greater than or equal to the threshold along with their indices.
    filtered_alphas = [(value, idx) for idx, value in enumerate(alphas) if value >= th]
    
    if not filtered_alphas:
        # If there are no matching values, return None and an empty list.
        return 0, []
    
    # Find the maximum value and all its corresponding indices (if there are multiple occurrences of the same maximum value).
    max_alpha = max(filtered_alphas, key=lambda x: x[0])[0]
    indices = [idx for value, idx in filtered_alphas if value > th]
    
    # Step 4: Return the maximum value and its index
    return max_alpha, indices


class StableDiffuser(torch.nn.Module):

    def __init__(
        self,
        scheduler="LMS",
        cache_dir="/opt/data/private/hugging_face",  # 自定义缓存路径
        concepts_to_erase = None,
        neutral_concept = "",
        params={"gamma":0.02, "beta":-0.06, "alpha_f":None, "erase_index_f":None,
                "lambda":1.0, "alpha_threshold":0.5, "detect_threshold":0.5},
        detect_method=None
    ):

        super().__init__()

        # Ensure the directory exists and has appropriate permissions.
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.concepts_to_erase = concepts_to_erase
        self.neutral_concept = neutral_concept
        self.params = params
        self.detect_method = detect_method
        self.version = 5

        try:
            # Load models with error handling and logging.
            print("Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="vae", cache_dir=cache_dir
            )

            print("Loading tokenizer and text encoder...")
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14", cache_dir=cache_dir
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14", cache_dir=cache_dir
            )

            print("Loading UNet model...")
            self.unet = UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="unet", cache_dir=cache_dir
            )

            print("Loading feature extractor and safety checker...")
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="feature_extractor",
                cache_dir=cache_dir,
            )
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="safety_checker",
                cache_dir=cache_dir,
            )

            print("Setting up scheduler...")
            if scheduler == "LMS":
                self.scheduler = LMSDiscreteScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    num_train_timesteps=1000,
                )
            elif scheduler == "DDIM":
                self.scheduler = DDIMScheduler.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    subfolder="scheduler",
                    cache_dir=cache_dir,
                )
            elif scheduler == "DDPM":
                self.scheduler = DDPMScheduler.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    subfolder="scheduler",
                    cache_dir=cache_dir,
                )

            self.eval()
            print("All components loaded successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise
    
    def reset_params(self, params:dict):
        for key in params.keys():
            assert key in self.params, "key is not in the params list!"
            self.params[key] = params[key]
    
    def get_noise(self, batch_size, img_size, generator=None):

        param = list(self.parameters())[0]

        return (
            torch.randn(
                (batch_size, self.unet.in_channels, img_size // 8, img_size // 8),
                generator=generator,
            )
            .type(param.dtype)
            .to(param.device)
        )

    def add_noise(self, latents, noise, step):

        return self.scheduler.add_noise(
            latents, noise, torch.tensor([self.scheduler.timesteps[step]])
        )

    def text_tokenize(self, prompts):

        return self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def text_detokenize(self, tokens):

        return [
            self.tokenizer.decode(token)
            for token in tokens
            if token != self.tokenizer.vocab_size - 1
        ]

    def text_encode(self, tokens):

        return self.text_encoder(tokens.input_ids.to(self.unet.device))[0]

    def decode(self, latents):

        return self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample

    def encode(self, tensors):

        return self.vae.encode(tensors).latent_dist.mode() * 0.18215

    def to_image(self, image):

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def set_scheduler_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.unet.device)

    def get_initial_latents(self, n_imgs, img_size, n_prompts, generator=None):

        noise = self.get_noise(n_imgs, img_size, generator=generator).repeat(
            n_prompts, 1, 1, 1
        )

        latents = noise * self.scheduler.init_noise_sigma

        return latents

    def get_text_embeddings(self, prompts, n_imgs):

        text_tokens = self.text_tokenize(prompts)

        text_embeddings = self.text_encode(text_tokens)

        unconditional_tokens = self.text_tokenize([""] * len(prompts))

        unconditional_embeddings = self.text_encode(unconditional_tokens)

        text_embeddings = torch.cat(
            [unconditional_embeddings, text_embeddings]
        ).repeat_interleave(n_imgs, dim=0)

        return text_embeddings

    def get_multi_erased_embedding(self, text_in, text_src, text_n, n_imgs,
                                   gamma=0.02, beta=-0.12, alpha_f=None, erase_index_f=None,
                                   show_alpha=True):
        if erase_index_f is None:
            erase_index_f = []

        embedding_in = self.get_text_embeddings(text_in, n_imgs)
        embedding_n = self.get_text_embeddings([text_n], n_imgs)
                                    
        if isinstance(text_src, str):
            embedding_src = [self.get_text_embeddings([text_src], n_imgs)]
        elif isinstance(text_src, list) or isinstance(text_src, tuple):
            embedding_src = [self.get_text_embeddings([text_s], n_imgs) for text_s in text_src]
        
        s = [compute_similarity(embedding_s - embedding_n, embedding_in) for embedding_s in embedding_src]
        embedding_alpha_s = [sigmoid_kernel(s_i, gamma, beta).unsqueeze(1).unsqueeze(1) for s_i in s]

        alpha_final, indexs = get_selected_alpha(embedding_alpha_s, self.params["alpha_threshold"])
        if alpha_f is not None:
            alpha_final = alpha_f

        indexs = list(set(indexs) | set(erase_index_f))
        selected_concepts = [text_src[index] for index in indexs]

        if show_alpha:
            print(f"alpha: {alpha_final}, index:{indexs}")
            print("Concepts semantically detected: " + ", ".join(selected_concepts))

        embedding_to_erase = self.get_text_embeddings([", ".join(selected_concepts)], n_imgs)
        
        embedding_output = embedding_in - alpha_final * (embedding_to_erase - embedding_n)

#         if alpha_final != 0:
#             embedding_output = embedding_output * embedding_n.norm(2) / embedding_output.norm(2)

        return embedding_output, alpha_final, indexs
    
    def get_similarity_between_prompts(self, text_in, text_src, text_n):
        embedding_in = self.get_text_embeddings(text_in, 1)
        embedding_n = self.get_text_embeddings([text_n], 1)
        
        if isinstance(text_src, str):
            embedding_src = [self.get_text_embeddings([text_src], 1)]
        elif isinstance(text_src, list) or isinstance(text_src, tuple):
            embedding_src = [self.get_text_embeddings([text_s], 1) for text_s in text_src]
            
        s = [compute_similarity(embedding_s - embedding_n, embedding_in) for embedding_s in embedding_src]
        print(s)
        

    def predict_noise(self, iteration, latents, text_embeddings, guidance_scale=7.5):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latents = torch.cat([latents] * 2)
        latents = self.scheduler.scale_model_input(
            latents, self.scheduler.timesteps[iteration]
        )

        # predict the noise residual
        noise_prediction = self.unet(
            latents,
            self.scheduler.timesteps[iteration],
            encoder_hidden_states=text_embeddings,
        ).sample

        # perform guidance
        noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
        noise_prediction = noise_prediction_uncond + guidance_scale * (
            noise_prediction_text - noise_prediction_uncond
        )

        return noise_prediction

    @torch.no_grad()
    def diffusion(
        self,
        latents,
        text_embeddings,
        end_iteration=1000,
        start_iteration=0,
        return_steps=False,
        pred_x0=False,
        trace_args=None,
        show_progress=True,
        **kwargs,
    ):

        latents_steps = []
        trace_steps = []

        trace = None

        for iteration in tqdm(
            range(start_iteration, end_iteration), disable=not show_progress
        ):

            if trace_args:

                trace = TraceDict(self, **trace_args)

            noise_pred = self.predict_noise(
                iteration, latents, text_embeddings, **kwargs
            )

            # compute the previous noisy sample x_t -> x_t-1
            output = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[iteration], latents
            )

            if trace_args:

                trace.close()

                trace_steps.append(trace)

            latents = output.prev_sample

            if return_steps or iteration == end_iteration - 1:

                output = output.pred_original_sample if pred_x0 else latents

                if return_steps:
                    latents_steps.append(output.cpu())
                else:
                    latents_steps.append(output)

        return latents_steps, trace_steps

    @torch.no_grad()
    def __call__(
        self,
        prompts,
        img_size=512,
        n_steps=50,
        n_imgs=1,
        end_iteration=None,
        generator=None,
        show_alpha=True,
        use_safety_checker=True,
        alpha_f=None,
        **kwargs,
    ):

        if not isinstance(prompts, list):

            prompts = [prompts]

        if self.concepts_to_erase is None:
            self.concepts_to_erase = ""
        
        # first stage inference
        text_embeddings, alpha1st, indexs1st = self.get_multi_erased_embedding(prompts, 
                                                          self.concepts_to_erase, 
                                                          self.neutral_concept, 
                                                          n_imgs,
                                                          gamma=self.params["gamma"], 
                                                          beta=self.params["beta"], 
                                                          alpha_f=self.params["alpha_f"], 
                                                          erase_index_f=self.params["erase_index_f"],
                                                          show_alpha=show_alpha)

        images_steps = self.call_with_embedding(text_embeddings,
                                                img_size=img_size,
                                                n_steps=n_steps,
                                                n_imgs=n_imgs,
                                                prompts_len=len(prompts),
                                                end_iteration=end_iteration,
                                                generator=generator,
                                                use_safety_checker=use_safety_checker,
                                                **kwargs
                                                )
        
        alpha_feed_back, indexs = self.visual_detection_feedback(images_steps, self.params, show_alpha)
        indexs_union = list(set(indexs1st) | set(indexs))

        # second stage inference
        if alpha_feed_back >= self.params["detect_threshold"]:
            alpha_feed_back = alpha_feed_back * self.params["lambda"]
            alpha_feed_back = max(alpha_feed_back, alpha1st)
            text_embeddings, alpha2st, indexs2st = self.get_multi_erased_embedding(prompts, 
                                                            self.concepts_to_erase, 
                                                            self.neutral_concept, 
                                                            n_imgs,
                                                            gamma=self.params["gamma"], 
                                                            beta=self.params["beta"], 
                                                            alpha_f=alpha_feed_back, 
                                                            erase_index_f=indexs_union,
                                                            show_alpha=show_alpha)
            images_steps = self.call_with_embedding(text_embeddings,
                                                img_size=img_size,
                                                n_steps=n_steps,
                                                n_imgs=n_imgs,
                                                prompts_len=len(prompts),
                                                end_iteration=end_iteration,
                                                generator=generator,
                                                use_safety_checker=use_safety_checker,
                                                **kwargs
                                                )
            print(alpha2st)

        return images_steps
    
    
    @torch.no_grad()
    def call_with_embedding(self,
                            embedding,
                            img_size=512,
                            n_steps=50,
                            n_imgs=1,
                            prompts_len=1,
                            end_iteration=None,
                            generator=None,
                            use_safety_checker=True,
                            **kwargs
                            ):
        
        assert 0 <= n_steps <= 1000
        
        self.set_scheduler_timesteps(n_steps)
        
        latents = self.get_initial_latents(
            n_imgs, img_size, prompts_len, generator=generator
        )
        
        end_iteration = end_iteration or n_steps
        
        latents_steps, trace_steps = self.diffusion(
            latents, embedding, end_iteration=end_iteration, **kwargs
        )
        
        latents_steps = [
            self.decode(latents.to(self.unet.device)) for latents in latents_steps
        ]
        images_steps = [self.to_image(latents) for latents in latents_steps]
            
        for i in range(len(images_steps)):
            if use_safety_checker:
                self.safety_checker = self.safety_checker.float()
                safety_checker_input = self.feature_extractor(
                    images_steps[i], return_tensors="pt"
                ).to(latents_steps[0].device)
                image, has_nsfw_concept = self.safety_checker(
                    images=latents_steps[i].float().cpu().numpy(),
                    clip_input=safety_checker_input.pixel_values.float(),
                )
            else:
                image = latents_steps[i].float().cpu().numpy()

            images_steps[i][0] = self.to_image(torch.from_numpy(image))[0]

        images_steps = list(zip(*images_steps))

        if trace_steps:

            return images_steps, trace_steps

        return images_steps
    
    def visual_detection_feedback(self, images_steps, params, show_alpha):
        if self.detect_method is None:
            self.detect_method = lambda images, params: (0, [])
        alpha_feed_back, indexs = self.detect_method(images_steps, params)
        if show_alpha:
            print(f"visual alpha: {alpha_feed_back}, concepts index:{indexs}")

        return alpha_feed_back, indexs
