import os
import comfy
import torch
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageEnhance, ImageOps, ImageDraw
import random
from typing import Dict, List, Optional
from .fetch_models import LLMService, get_available_models, validate_model

import hashlib
from functools import lru_cache

# Utility functions for image conversion
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class MinxMergeNode:
    def __init__(self):
        self.last_seed = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "llm_service": ([LLMService.OPENAI.value, LLMService.ANTHROPIC.value], {"default": LLMService.OPENAI.value}),
                "max_tokens": ("INT", {"default": 100, "min": 1, "max": 4096}),
                "openai_model": (get_available_models(LLMService.OPENAI), {"default": "none"}),
                "anthropic_model": (get_available_models(LLMService.ANTHROPIC), {"default": "none"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "debug_mode": ("BOOLEAN", {"default": False}),
                "instruction": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "example": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "seed")
    FUNCTION = "generate_prompt"
    CATEGORY = "Minx Merge"

    def generate_prompt(self, image: torch.Tensor, llm_service: str, max_tokens: int,
                        openai_model: str, anthropic_model: str,
                        temperature: float, seed: int, debug_mode: bool,
                        instruction: str, example: str, prompt: str) -> tuple:
        import logging
        logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
        logger = logging.getLogger(__name__)

        try:
            logger.debug(f"Starting prompt generation with {llm_service} service")
            service = LLMService(llm_service)

            # Handle seed - ensure it stays within ComfyUI's 32-bit range
            if self.last_seed is not None:
                # Increment seed but wrap around if it exceeds 32-bit limit
                incremented_seed = self.last_seed + 1
                if incremented_seed > 0xffffffff:  # 2^32 - 1
                    seed = 0  # Wrap around to 0
                else:
                    seed = incremented_seed
            self.last_seed = seed

            np.random.seed(seed)
            random.seed(seed)
            # Set the seed
            torch.manual_seed(seed)

            # Select the appropriate model based on the chosen service
            if service == LLMService.OPENAI:
                model = openai_model
            elif service == LLMService.ANTHROPIC:
                model = anthropic_model
            else:
                raise ValueError(f"Unsupported LLM service: {llm_service}")

            logger.debug(f"Using model: {model}")

            if model == "none":
                raise ValueError(f"No model selected for {service.value}")

            # Get API key from environment
            api_key = self._get_api_key(service)
            logger.debug(f"API key retrieved for {service.value}")

            # Validate API key
            self._validate_api_key(api_key, service)

            # Validate model
            if not validate_model(service, model):
                raise ValueError(f"Invalid model {model} for service {service.value}")

            # Process image
            logger.debug("Processing image")
            image_data = self._process_image(image)

            # Build the user prompt from instruction, example, and prompt
            user_prompt = ""
            if instruction:
                user_prompt += f"Instruction: {instruction}\n"
            if example:
                user_prompt += f"Example: {example}\n"
            if prompt:
                user_prompt += f"Prompt: {prompt}"

            logger.debug("User prompt generated")

            # Log the size of the user prompt
            prompt_size = len(user_prompt.encode('utf-8'))
            logger.debug(f"User prompt size: {prompt_size} bytes")

            # Call appropriate LLM API with retry mechanism
            logger.debug(f"Calling {service.value} API")
            result = self._call_api_with_retry(
                service, api_key, model, user_prompt, image_data, max_tokens, temperature
            )

            logger.info(f"Generated prompt: {result}")
            return (result, seed)
        except Exception as e:
            error_message = f"Error generating prompt: {str(e)}"
            logger.error(error_message)
            return (error_message, seed)

    def _call_anthropic(self, api_key: str, model: str, user_prompt: str, image_data: str, max_tokens: int, temperature: float) -> str:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data.split(",")[1]}},
                    ],
                }
            ]
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return response.content[0].text.strip()
        except Exception as e:
            if "too many total text bytes" in str(e):
                raise RuntimeError(f"Anthropic API request too large. Please reduce input size.")
            else:
                raise RuntimeError(f"Anthropic API error: {str(e)}")

    def _get_api_key(self, service: LLMService) -> str:
        key = os.getenv(f"{service.value.upper()}_API_KEY")
        if not key:
            raise ValueError(f"Please set {service.value.upper()}_API_KEY environment variable")
        return key

    def _validate_api_key(self, api_key: str, service: LLMService) -> None:
        if not api_key:
            raise ValueError(f"API key for {service.value} is missing or invalid")

    def _validate_input(self, art_style: str, max_tokens: int) -> None:
        if not art_style.strip():
            raise ValueError("Art style cannot be empty")
        if max_tokens < 1 or max_tokens > 4096:
            raise ValueError("max_tokens must be between 1 and 4096")

    def _process_image(self, image: torch.Tensor, max_size: int = 512) -> str:
        import base64
        from io import BytesIO

        # Convert tensor to PIL Image
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        
        # Resize image if it's too large
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"

    def _truncate_art_style(self, art_style: str, max_length: int = 1000) -> str:
        if len(art_style) <= max_length:
            return art_style
        return art_style[:max_length] + "..."

    def _call_api_with_retry(self, service: LLMService, api_key: str, model: str, user_prompt: str, image_data: str, max_tokens: int, temperature: float, max_retries: int = 3) -> str:
        import time
        for attempt in range(max_retries):
            try:
                if service == LLMService.OPENAI:
                    return self._call_openai(api_key, model, user_prompt, image_data, max_tokens, temperature)
                elif service == LLMService.ANTHROPIC:
                    return self._call_anthropic(api_key, model, user_prompt, image_data, max_tokens, temperature)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        raise RuntimeError("Max retries reached")

    def _call_openai(self, api_key: str, model: str, user_prompt: str, image_data: str, max_tokens: int, temperature: float) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        try:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]}
            ]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                raise RuntimeError(f"OpenAI API rate limit exceeded. Please try again later.")
            raise RuntimeError(f"OpenAI API error: {str(e)}")

class ImageRotateNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["internal", "external"], {"default": "internal"}),
                "rotation": ("INT", {"default": 90, "min": 0, "max": 360, "step": 90}),
                "sampler": (["nearest", "bicubic", "bilinear"], {"default": "bilinear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_rotate"
    CATEGORY = "Minx Merge"

    def image_rotate(self, images, mode, rotation, sampler):
        batch_tensor = []
        for image in images:
            # Convert to PIL Image
            image = tensor2pil(image)

            # Check rotation
            if rotation > 360:
                rotation = int(360)
            if (rotation % 90 != 0):
                rotation = int((rotation//90)*90)

            # Set Sampler
            if sampler:
                if sampler == 'nearest':
                    resampling = Image.NEAREST
                elif sampler == 'bicubic':
                    resampling = Image.BICUBIC
                elif sampler == 'bilinear':
                    resampling = Image.BILINEAR
                else:
                    resampling = Image.BILINEAR

            # Rotate Image
            if mode == 'internal':
                image = image.rotate(rotation, resampling)
            else:
                rot = int(rotation / 90)
                for _ in range(rot):
                    # Image.ROTATE_90 = 2 (transpose method constant for 90-degree rotation)
                    image = image.transpose(2)

            batch_tensor.append(pil2tensor(image))

        batch_tensor = torch.cat(batch_tensor, dim=0)

        return (batch_tensor, )

class ChromaticAberrationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "red_offset": ("INT", {"default": 2, "min": -50, "max": 50, "step": 1}),
                "green_offset": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "blue_offset": ("INT", {"default": -2, "min": -50, "max": 50, "step": 1}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fade_radius": ("INT", {"default": 12, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_chromatic_aberration"
    CATEGORY = "Minx Merge/Image/Filter"

    def apply_chromatic_aberration(self, image, red_offset=4, green_offset=2, blue_offset=0, intensity=1, fade_radius=12):
        batch_tensor = []

        for img in image:
            img_pil = tensor2pil(img)
            result = self._chromatic_aberration_effect(img_pil, red_offset, green_offset, blue_offset, intensity, fade_radius)
            batch_tensor.append(pil2tensor(result))

        return (torch.cat(batch_tensor, dim=0),)

    def _chromatic_aberration_effect(self, img, r_offset, g_offset, b_offset, intensity, fade_radius):
        def lingrad(size, direction, white_ratio):
            image = Image.new('RGB', size)
            draw = ImageDraw.Draw(image)
            if direction == 'vertical':
                black_end = size[1] - white_ratio
                range_start = 0
                range_end = size[1]
                range_step = 1
                for y in range(range_start, range_end, range_step):
                    color_ratio = y / size[1]
                    if y <= black_end:
                        color = (0, 0, 0)
                    else:
                        color_value = int(((y - black_end) / (size[1] - black_end)) * 255)
                        color = (color_value, color_value, color_value)
                    draw.line([(0, y), (size[0], y)], fill=color)
            elif direction == 'horizontal':
                black_end = size[0] - white_ratio
                range_start = 0
                range_end = size[0]
                range_step = 1
                for x in range(range_start, range_end, range_step):
                    color_ratio = x / size[0]
                    if x <= black_end:
                        color = (0, 0, 0)
                    else:
                        color_value = int(((x - black_end) / (size[0] - black_end)) * 255)
                        color = (color_value, color_value, color_value)
                    draw.line([(x, 0), (x, size[1])], fill=color)

            return image.convert("L")

        def create_fade_mask(size, fade_radius):
            mask = Image.new("L", size, 255)

            left = ImageOps.invert(lingrad(size, 'horizontal', int(fade_radius * 2)))
            right = left.copy().transpose(Image.FLIP_LEFT_RIGHT)
            top = ImageOps.invert(lingrad(size, 'vertical', int(fade_radius * 2)))
            bottom = top.copy().transpose(Image.FLIP_TOP_BOTTOM)

            # Multiply masks with the original mask image
            mask = ImageChops.multiply(mask, left)
            mask = ImageChops.multiply(mask, right)
            mask = ImageChops.multiply(mask, top)
            mask = ImageChops.multiply(mask, bottom)
            mask = ImageChops.multiply(mask, mask)

            return mask

        # split the channels of the image
        r, g, b = img.split()

        # apply the offset to each channel (chromatic aberration typically offsets red/blue horizontally)
        r_offset_img = ImageChops.offset(r, r_offset, 0)
        g_offset_img = ImageChops.offset(g, green_offset, 0)  
        b_offset_img = ImageChops.offset(b, b_offset, 0)

        # merge the channels with the offsets
        merged = Image.merge("RGB", (r_offset_img, g_offset_img, b_offset_img))

        # create fade masks for blending
        fade_mask = create_fade_mask(img.size, fade_radius)
        
        # blend the original image with the chromatic aberration effect using the fade mask
        result = Image.composite(merged, img, fade_mask)
        
        # apply intensity blending
        final_result = Image.blend(img, result, intensity)
        
        return final_result

class FilmGrainNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "density": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "intensity": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 1.0, "step": 0.01}),
                "highlights": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 255.0, "step": 0.01}),
                "supersample_factor": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "device": (["AUTO", "CPU", "GPU"], {"default": "AUTO"})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_film_grain"
    CATEGORY = "Minx Merge/Image/Filter"

    def apply_film_grain(self, image, density=0.5, intensity=0.6, highlights=1.0, supersample_factor=4, device="AUTO"):
        batch_tensor = []

        for img in image:
            img_pil = tensor2pil(img)
            result = self._film_grain_effect(img_pil, density, intensity, highlights, supersample_factor, device)
            batch_tensor.append(pil2tensor(result))

        return (torch.cat(batch_tensor, dim=0),)

    def _film_grain_effect(self, img, density=0.1, intensity=1.0, highlights=1.0, supersample_factor=4, device="AUTO"):
        """
        Apply grayscale noise with specified density, intensity, and highlights to a PIL image.
        Device parameter is available for future GPU-accelerated implementations.
        """
        # Current implementation uses PIL/CPU regardless of device setting
        # This parameter is prepared for future GPU acceleration
        
        img_gray = img.convert('L')
        original_size = img.size
        img_gray = img_gray.resize(
            ((img.size[0] * supersample_factor), (img.size[1] * supersample_factor)), Image.Resampling.LANCZOS)
        num_pixels = int(density * img_gray.size[0] * img_gray.size[1])

        noise_pixels = []
        for i in range(num_pixels):
            x = random.randint(0, img_gray.size[0]-1)
            y = random.randint(0, img_gray.size[1]-1)
            noise_pixels.append((x, y))

        for x, y in noise_pixels:
            value = random.randint(0, 255)
            img_gray.putpixel((x, y), value)

        img_noise = img_gray.convert('RGB')
        img_noise = img_noise.filter(ImageFilter.GaussianBlur(radius=0.125))
        img_noise = img_noise.resize(original_size, Image.Resampling.LANCZOS)
        img_noise = img_noise.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img_final = Image.blend(img, img_noise, intensity)
        enhancer = ImageEnhance.Brightness(img_final)
        img_highlights = enhancer.enhance(highlights)

        # Return the final image
        return img_highlights

# Import LaMa node
from .lama_remove_object import LamaRemoveObject

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MinxMergeNode": MinxMergeNode,
    "ImageRotateNode": ImageRotateNode,
    "ChromaticAberrationNode": ChromaticAberrationNode,
    "FilmGrainNode": FilmGrainNode,
    "LamaRemoveObject": LamaRemoveObject
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinxMergeNode": "Minx Merge",
    "ImageRotateNode": "Image Rotate",
    "ChromaticAberrationNode": "Chromatic Aberration",
    "FilmGrainNode": "Film Grain",
    "LamaRemoveObject": "LaMa Remove Object"
}