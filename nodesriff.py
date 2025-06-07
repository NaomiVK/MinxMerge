import os
import torch
from typing import List
from .fetch_models import LLMService, get_available_models, validate_model

class AutoRiffingPromptNode:
    def __init__(self):
        self.last_seed = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_prompt": ("STRING", {"multiline": True}),
                "instructions": ("STRING", {"multiline": True, "forceInput": True}),
                "iterations": ("INT", {"default": 3, "min": 1, "max": 10}),
                "creativity_level": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "target_creativity": ("INT", {"default": 8, "min": 1, "max": 10}),
                "target_coherence": ("INT", {"default": 8, "min": 1, "max": 10}),
                "target_relevance": ("INT", {"default": 8, "min": 1, "max": 10}),
                "max_tokens": ("INT", {"default": 300, "min": 1, "max": 4096}),
                "llm_service": ([LLMService.OPENAI.value, LLMService.ANTHROPIC.value], {"default": LLMService.OPENAI.value}),
                "openai_model": (get_available_models(LLMService.OPENAI), {"default": "none"}),
                "anthropic_model": (get_available_models(LLMService.ANTHROPIC), {"default": "none"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "seed", "llm_service", "model")
    FUNCTION = "auto_riff_prompt"
    CATEGORY = "Minx Merge"

    def auto_riff_prompt(self, initial_prompt: str, instructions: str, iterations: int, creativity_level: float,
                         target_creativity: int, target_coherence: int, target_relevance: int,
                         max_tokens: int, llm_service: str, openai_model: str, anthropic_model: str,
                         seed: int, debug_mode: bool) -> tuple:
        import logging
        logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
        logger = logging.getLogger(__name__)

        try:
            logger.debug(f"Starting auto-riffing with {llm_service} service")
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

            prompt = initial_prompt
            for i in range(iterations):
                logger.debug(f"Iteration {i + 1}")
                logger.debug(f"Current prompt: {prompt}")
                
                logger.debug(f"Calling API with prompt: {prompt}")
                result = self._call_api_with_retry(service, api_key, model, instructions, prompt, max_tokens, creativity_level)
                logger.debug(f"API response: {result}")
                
                # Parse the result
                lines = result.split('\n')
                refined_prompt = lines[0] if lines else ""
                scores = self._parse_scores(lines[1:])
                
                logger.debug(f"Iteration {i + 1}: Creativity={scores[0]}, Coherence={scores[1]}, Relevance={scores[2]}")
                logger.debug(f"Refined Prompt: {refined_prompt}")

                # Stop refining if the scores are high enough
                if (scores[0] >= target_creativity and scores[1] >= target_coherence and scores[2] >= target_relevance):
                    logger.debug(f"Stopping refinement due to high scores: Creativity={scores[0]}, Coherence={scores[1]}, Relevance={scores[2]}")
                    break

                if refined_prompt:
                    prompt = refined_prompt
                else:
                    logger.warning("Received empty refined prompt, using previous prompt")

            logger.info(f"Final prompt: {prompt}")
            return (prompt, seed, llm_service, model)
        except Exception as e:
            error_message = f"Error in auto-riffing: {str(e)}"
            logger.error(error_message)
            return (error_message, seed, llm_service, model)

    def _get_api_key(self, service: LLMService) -> str:
        key = os.getenv(f"{service.value.upper()}_API_KEY")
        if not key:
            raise ValueError(f"Please set {service.value.upper()}_API_KEY environment variable")
        return key

    def _validate_api_key(self, api_key: str, service: LLMService) -> None:
        if not api_key:
            raise ValueError(f"API key for {service.value} is missing or invalid")

    def _parse_scores(self, lines: List[str]) -> List[int]:
        scores = []
        for line in lines:
            try:
                score = int(line.split(':')[-1].strip())
                scores.append(score)
            except ValueError:
                # If we can't parse a score, append a default value
                scores.append(5)
        
        # Ensure we always return 3 scores
        while len(scores) < 3:
            scores.append(5)
        
        return scores[:3]

    def _call_api_with_retry(self, service: LLMService, api_key: str, model: str, instructions: str, user_prompt: str, max_tokens: int, temperature: float, max_retries: int = 3) -> str:
        import time
        for attempt in range(max_retries):
            try:
                if service == LLMService.OPENAI:
                    return self._call_openai(api_key, model, instructions, user_prompt, max_tokens, temperature)
                elif service == LLMService.ANTHROPIC:
                    return self._call_anthropic(api_key, model, instructions, user_prompt, max_tokens, temperature)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        raise RuntimeError("Max retries reached")

    def _call_openai(self, api_key: str, model: str, instructions: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        try:
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_prompt}
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

    def _call_anthropic(self, api_key: str, model: str, instructions: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        try:
            # Format the content to include both instructions and prompt
            formatted_content = f"{instructions}\n\nUser: {user_prompt}\n\nAssistant: Here's a refined version of your prompt:"
            
            # Build messages in Anthropic format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_content
                        }
                    ]
                }
            ]
            
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            return response.content[0].text.strip()
        except Exception as e:
            if "too many total text bytes" in str(e):
                raise RuntimeError(f"Anthropic API request too large. Please reduce input size.")
            raise RuntimeError(f"Anthropic API error: {str(e)}")

class EvaluationInstructionsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "evaluation_instructions": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instructions",)
    FUNCTION = "process"
    CATEGORY = "Minx Merge"

    def process(self, evaluation_instructions: str) -> tuple:
        return (evaluation_instructions,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AutoRiffingPromptNode": AutoRiffingPromptNode,
    "EvaluationInstructionsNode": EvaluationInstructionsNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoRiffingPromptNode": "Auto Riffing Prompt",
    "EvaluationInstructionsNode": "Evaluation Instructions"
}
