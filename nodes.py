import os
import comfy
import torch
import numpy as np
from PIL import Image
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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

            # Handle seed
            if self.last_seed is not None:
                seed = self.last_seed + 1
            self.last_seed = seed

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
                "mode": (["transpose", "internal"],),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 90}),
                "sampler": (["nearest", "bilinear", "bicubic"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "image_rotate"
    CATEGORY = "Minx Merge/Image/Transform"

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

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MinxMergeNode": MinxMergeNode,
    "ImageRotateNode": ImageRotateNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinxMergeNode": "Minx Merge",
    "ImageRotateNode": "Image Rotate"
}
