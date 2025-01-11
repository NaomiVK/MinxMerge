import os
import torch
from nodes import MinxMergeNode
from fetch_models import LLMService, get_available_models

def test_minx_merge_node():
    # Create an instance of MinxMergeNode
    node = MinxMergeNode()

    # Test input types
    input_types = MinxMergeNode.INPUT_TYPES()
    assert "art_style" in input_types["required"]
    assert "image1" in input_types["required"]
    assert "image2" in input_types["required"]
    assert "llm_service" in input_types["required"]
    assert "max_tokens" in input_types["required"]
    assert "system_prompt" in input_types["required"]
    assert "openai_model" in input_types["required"]
    assert "anthropic_model" in input_types["required"]
    assert "groq_model" in input_types["required"]

    print("Input types test passed.")

    # Test model fetching for each service
    for service in LLMService:
        models = get_available_models(service)
        print(f"Models for {service.value}: {models}")
        assert isinstance(models, list)
        assert "none" in models
        if service != LLMService.ANTHROPIC:  # Anthropic uses a predefined list
            assert len(models) > 1, f"No models fetched for {service.value}"
        if service == LLMService.GROQ:
            assert all("-vision-" in model.lower() for model in models if model != "none"), "Groq models should all contain '-vision-'"

    print("Model fetching test passed.")

    # Test generate_prompt method
    # Note: This is a basic test and won't actually call the LLM APIs
    try:
        result = node.generate_prompt(
            art_style="impressionist",
            image1=torch.rand(3, 64, 64),
            image2=torch.rand(3, 64, 64),
            llm_service=LLMService.OPENAI.value,
            max_tokens=100,
            system_prompt="Test prompt",
            openai_model="gpt-3.5-turbo",
            anthropic_model="none",
            groq_model="none"
        )
        print("Generate prompt test passed.")
    except Exception as e:
        print(f"Generate prompt test failed: {str(e)}")

if __name__ == "__main__":
    test_minx_merge_node()
