import os
import logging
from typing import Dict, List, Optional, Iterable
from enum import Enum
from abc import ABC, abstractmethod
import openai
from anthropic import Anthropic
from groq import Groq
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMService(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GROQ = "Groq"

class Model:
    def __init__(self, model_id):
        self.id = model_id

class ModelsContainer:
    def __init__(self, model_ids):
        self.data = [Model(model_id) for model_id in model_ids]

class ModelContainer:
    def __init__(self, models: List[str], service: LLMService):
        self._models = models
        self._service = service

    def get_models(self, sort_it: bool = True, with_none: bool = True, include_filter: Optional[Iterable[str]] = None, exclude_filter: Optional[Iterable[str]] = None) -> List[str]:
        models = ['none'] if with_none else []

        if include_filter and isinstance(include_filter, Iterable):
            filtered_models = [model for model in self._models 
                          if any(f.lower() in model.lower() for f in include_filter)]
        else:
            filtered_models = self._models[:]

        if exclude_filter and isinstance(exclude_filter, Iterable):
            filtered_models = [model for model in filtered_models 
                          if all(f.lower() not in model.lower() for f in exclude_filter)]

        models.extend(filtered_models)

        if sort_it:
            models.sort()

        return models

    @property
    def service(self) -> LLMService:
        return self._service

class ModelFetchStrategy(ABC):
    @abstractmethod
    def fetch_models(self, key: str, api_obj, **kwargs):
        pass

class OpenAIModelFetch(ModelFetchStrategy):
    def fetch_models(self, key: str, api_obj, **kwargs):
        if not key:
            logging.warning("No OpenAI Key found.")
            return None

        try:
            client = api_obj.OpenAI(api_key=key)  # <-- fixed this line
            models = client.models.list()
            return ModelsContainer([model.id for model in models.data if "gpt" in model.id.lower()])
        except Exception as e:
            logging.error(f"OpenAI Key is invalid or missing, unable to generate list of models. Error: {e}")
            return None


class AnthropicModelFetch(ModelFetchStrategy):
    def fetch_models(self, key: str, api_obj, **kwargs):
        # Anthropic doesn't have a list_models endpoint, so we'll use a predefined list
        models = [
            'claude-3-haiku-20240307',
            'claude-3-5-haiku-latest',
            'claude-3-sonnet-20240229',
            'claude-3-5-sonnet-20240620',
            'claude-3-5-sonnet-latest',
            'claude-3-opus-20240229'
        ]
        return ModelsContainer(models)

class GroqModelFetch(ModelFetchStrategy):
    def fetch_models(self, key: str, api_obj, **kwargs):
        if not key:
            logging.warning("No Groq Key found.")
            return None
        
        client = api_obj(api_key=key)
        try:
            model_list = client.models.list()
            vision_models = [model for model in model_list.data if self.is_vision_model(model.id)]
            if not vision_models:
                logging.warning("No Groq vision models found. This might be due to API changes or limited access.")
            return ModelsContainer([model.id for model in vision_models])
        except Exception as e:
            logging.error(f"Groq Key is invalid or missing, unable to generate list of models. Error: {e}")
            return None

    def is_vision_model(self, model_id: str) -> bool:
        return "-vision-" in model_id.lower()

class ModelFetcher:
    def __init__(self):
        self.strategies = {
            LLMService.OPENAI: OpenAIModelFetch(),
            LLMService.ANTHROPIC: AnthropicModelFetch(),
            LLMService.GROQ: GroqModelFetch(),
        }

    def fetch_models(self, service: LLMService, key: str, api_obj: object = None, **kwargs):
        strategy = self.strategies.get(service)
        if not strategy:
            logging.error(f"Unsupported service: {service}")
            raise ValueError(f"Unsupported service: {service}")
        return strategy.fetch_models(key, api_obj, **kwargs)

from datetime import datetime, timedelta

model_cache = {}
CACHE_DURATION = timedelta(hours=1)

def get_available_models(service: LLMService) -> List[str]:
    current_time = datetime.now()
    if service in model_cache and current_time - model_cache[service]['timestamp'] < CACHE_DURATION:
        return model_cache[service]['models']

    api_key = os.getenv(f"{service.value.upper()}_API_KEY")
    if not api_key:
        logging.warning(f"No API key found for {service.value}")
        return []
    try:
        fetcher = ModelFetcher()
        if service == LLMService.OPENAI:
            models = fetcher.fetch_models(service, api_key, openai)
        elif service == LLMService.ANTHROPIC:
            models = fetcher.fetch_models(service, api_key)
        elif service == LLMService.GROQ:
            models = fetcher.fetch_models(service, api_key, Groq)
        else:
            models = None

        if models and hasattr(models, 'data'):
            model_list = [model.id for model in models.data]
            available_models = ModelContainer(model_list, service).get_models()
            model_cache[service] = {
                'models': available_models,
                'timestamp': current_time
            }
            return available_models
        else:
            logging.warning(f"No models found for {service.value}")
            return []
    except Exception as e:
        logging.error(f"Error fetching models for {service.value}: {str(e)}")
        return []

def validate_model(service: LLMService, model: str) -> bool:
    return model in get_available_models(service)
