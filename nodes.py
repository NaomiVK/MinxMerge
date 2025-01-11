import os
import comfy
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
from .fetch_models import LLMService, get_available_models, validate_model

import hashlib
from functools import lru_cache

class MinxMergeNode:
    def __init__(self):
        self.last_seed = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "art_style": ("STRING", {"default": "", "multiline": True}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "llm_service": ([LLMService.OPENAI.value, LLMService.ANTHROPIC.value], {"default": LLMService.OPENAI.value}),
                "max_tokens": ("INT", {"default": 100, "min": 1, "max": 4096}),
                "system_prompt": ("STRING", {
                    "default": """Create imaginative prompts by integrating elements from two images into one cohesive and fully integrated scene. Ensure the description is concise, clear, and reflects the seamless fusion of both images. Users will specify an art style, and should be shortened to include the most important art style traits. The description of the images should avoid terms like 'blend' or 'merge,' and be capped at 120 words.

# Steps

1. Seamlessly integrate elements from both images into a single imaginative scene.
2. Avoid using terms related to merging or blending to describe the scene.
3. Ensure the final description is creative, coherent, and comprehensible within the 100 word limit.

# Output Format

- Begin with the user supplied art style.
- Follow with a unified scene description that reflects the integration of elements from both images.
- The entire output must be no longer than 120 words.""",
                    "multiline": True
                }),
                "openai_model": (get_available_models(LLMService.OPENAI), {"default": "none"}),
                "anthropic_model": (get_available_models(LLMService.ANTHROPIC), {"default": "none"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "seed")
    FUNCTION = "generate_prompt"
    CATEGORY = "Minx Merge"

    def generate_prompt(self, art_style: str, image1: torch.Tensor, image2: torch.Tensor, 
                        llm_service: str, max_tokens: int, system_prompt: str, 
                        openai_model: str, anthropic_model: str, 
                        temperature: float, seed: int, debug_mode: bool) -> tuple:
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
            
            # Validate input parameters
            self._validate_input(art_style, max_tokens)
            
            # Process images
            logger.debug("Processing images")
            image1_data = self._process_image(image1)
            image2_data = self._process_image(image2)
            
            # Generate user prompt
            user_prompt = f"Art style: {self._truncate_art_style(art_style)}"
            logger.debug("User prompt generated")
            
            # Log the size of the user prompt
            prompt_size = len(user_prompt.encode('utf-8'))
            logger.debug(f"User prompt size: {prompt_size} bytes")
            
            # Call appropriate LLM API with retry mechanism
            logger.debug(f"Calling {service.value} API")
            result = self._call_api_with_retry(service, api_key, model, system_prompt, user_prompt, image1_data, image2_data, max_tokens, temperature)
            
            logger.info(f"Generated prompt: {result}")
            return (result, seed)
        except Exception as e:
            error_message = f"Error generating prompt: {str(e)}"
            logger.error(error_message)
            return (error_message, seed)

    def _validate_input(self, art_style: str, max_tokens: int) -> None:
        """Validate the input parameters."""
        if not art_style.strip():
            raise ValueError("Art style cannot be empty")
        if max_tokens < 1 or max_tokens > 4096:
            raise ValueError("max_tokens must be between 1 and 4096")

    def _validate_api_key(self, api_key: str, service: LLMService) -> None:
        """Validate the API key for the given service."""
        if not api_key:
            raise ValueError(f"API key for {service.value} is missing or invalid")
        # Add more specific validation logic for each service if needed

    def _generate_cache_key(self, service: LLMService, model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, seed: int, image1_data: str, image2_data: str) -> str:
        """Generate a unique cache key for the given inputs."""
        key_string = f"{service.value}:{model}:{system_prompt}:{user_prompt}:{max_tokens}:{temperature}:{seed}:{image1_data}:{image2_data}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _call_api_with_retry(self, service: LLMService, api_key: str, model: str, system_prompt: str, user_prompt: str, image1_data: str, image2_data: str, max_tokens: int, temperature: float, max_retries: int = 3) -> str:
        """Call the API with retry mechanism."""
        import time
        for attempt in range(max_retries):
            try:
                if service == LLMService.OPENAI:
                    return self._call_openai(api_key, model, system_prompt, user_prompt, image1_data, image2_data, max_tokens, temperature)
                elif service == LLMService.ANTHROPIC:
                    return self._call_anthropic(api_key, model, system_prompt, user_prompt, image1_data, image2_data, max_tokens, temperature)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        raise RuntimeError("Max retries reached")

    def _get_api_key(self, service: LLMService) -> str:
        key = os.getenv(f"{service.value.upper()}_API_KEY")
        if not key:
            raise ValueError(f"Please set {service.value.upper()}_API_KEY environment variable")
        return key

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

    def _check_request_size(self, prompt: str, max_size: int = 4000000) -> bool:
        """Check if the request size is within the allowed limit."""
        return len(prompt.encode('utf-8')) <= max_size

    def _call_openai(self, api_key: str, model: str, system_prompt: str, user_prompt: str, image1: str, image2: str, max_tokens: int, temperature: float) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image1}},
                    {"type": "image_url", "image_url": {"url": image2}}
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

    def _call_anthropic(self, api_key: str, model: str, system_prompt: str, user_prompt: str, image1: str, image2: str, max_tokens: int, temperature: float) -> str:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant: Here's a description based on the two images and the specified art style:"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image1.split(",")[1]}},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image2.split(",")[1]}},
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
            raise RuntimeError(f"Anthropic API error: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "MinxMergeNode": MinxMergeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MinxMergeNode": "Minx Merge",
}
