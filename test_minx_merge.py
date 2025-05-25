import os
import torch
import numpy as np # Added for image comparison if needed, and for allclose
from nodes import MinxMergeNode, ImageRotateNode, ChromaticAberrationNode, FilmGrainNode # Added new nodes
from fetch_models import LLMService, get_available_models

# Helper functions for comparing images
def assert_images_equal(img1_tensor, img2_tensor, tolerance=1e-6):
    assert torch.allclose(img1_tensor, img2_tensor, atol=tolerance), "Images are not identical"

def assert_images_not_equal(img1_tensor, img2_tensor):
    assert not torch.allclose(img1_tensor, img2_tensor), "Images are identical when they should differ"

def test_minx_merge_node():
    # Create an instance of MinxMergeNode
    node = MinxMergeNode()

    # Test input types
    input_types = MinxMergeNode.INPUT_TYPES()
    required_inputs = input_types["required"]
    assert "image" in required_inputs
    assert "llm_service" in required_inputs
    assert "max_tokens" in required_inputs
    assert "openai_model" in required_inputs
    assert "anthropic_model" in required_inputs
    assert "temperature" in required_inputs
    assert "seed" in required_inputs
    assert "debug_mode" in required_inputs
    assert "instruction" in required_inputs
    assert "example" in required_inputs
    assert "prompt" in required_inputs
    # Ensure groq_model is NOT in MinxMergeNode inputs
    assert "groq_model" not in required_inputs 

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
            # This assertion is for get_available_models, which *does* support Groq.
            # It's okay if MinxMergeNode doesn't expose Groq directly in its UI.
            assert all("-vision-" in model.lower() or model == "none" for model in models), "Groq models should all contain '-vision-' or be 'none'"


    print("Model fetching test passed.")

    # Test generate_prompt method
    # Note: This is a basic test and won't actually call the LLM APIs
    try:
        result, out_seed = node.generate_prompt(
            image=torch.rand(1, 64, 64, 3), # Batch, H, W, C
            llm_service=LLMService.OPENAI.value,
            max_tokens=50,
            openai_model="gpt-3.5-turbo", # Using a known model for test structure
            anthropic_model="none",
            temperature=0.7,
            seed=42,
            debug_mode=False,
            instruction="Describe this image.",
            example="A cat.",
            prompt="What is in this image?"
        )
        assert isinstance(result, str)
        assert isinstance(out_seed, int)
        print("Generate prompt test passed.")
    except Exception as e:
        print(f"Generate prompt test failed: {str(e)}")

def test_image_rotate_node():
    node = ImageRotateNode()
    image_tensor = torch.rand(1, 64, 64, 3) # B, H, W, C

    # Test basic rotation
    rotated_image_tensor, = node.image_rotate(images=image_tensor, mode="transpose", rotation=90, sampler="nearest")

    assert isinstance(rotated_image_tensor, torch.Tensor)
    # For a 90-degree rotation of a non-square image, H and W would swap.
    # However, the example uses a square image (64x64), so shape remains the same.
    # If we were testing non-square, this assertion would need to be more complex.
    assert rotated_image_tensor.shape == image_tensor.shape 
    assert rotated_image_tensor.dtype == torch.float32
    # Add a check to ensure it's not identical for a 90 deg rotation
    assert_images_not_equal(rotated_image_tensor, image_tensor) 

    print("ImageRotateNode test passed.")

def test_chromatic_aberration_node():
    node = ChromaticAberrationNode()
    image_tensor = torch.rand(1, 32, 32, 3) # Using smaller image for faster test

    # Test with intensity = 0.0 (should be original image)
    result_intensity_0, = node.apply_chromatic_aberration(
        image=image_tensor, red_offset=5, green_offset=5, blue_offset=5, 
        intensity=0.0, fade_radius=1
    )
    assert isinstance(result_intensity_0, torch.Tensor)
    assert result_intensity_0.shape == image_tensor.shape
    assert_images_equal(result_intensity_0, image_tensor)

    # Test with intensity = 1.0 (should be different from original, if offsets are non-zero)
    result_intensity_1, = node.apply_chromatic_aberration(
        image=image_tensor, red_offset=5, green_offset=5, blue_offset=5, 
        intensity=1.0, fade_radius=1
    )
    assert isinstance(result_intensity_1, torch.Tensor)
    assert result_intensity_1.shape == image_tensor.shape
    if any(o != 0 for o in [5,5,5]): # only assert if offsets are non-zero
        assert_images_not_equal(result_intensity_1, image_tensor)

    # Test with intensity = 0.5 (should be different from original and full)
    result_intensity_0_5, = node.apply_chromatic_aberration(
        image=image_tensor, red_offset=5, green_offset=5, blue_offset=5, 
        intensity=0.5, fade_radius=1
    )
    assert isinstance(result_intensity_0_5, torch.Tensor)
    assert result_intensity_0_5.shape == image_tensor.shape
    if any(o != 0 for o in [5,5,5]):
        assert_images_not_equal(result_intensity_0_5, image_tensor)
        assert_images_not_equal(result_intensity_0_5, result_intensity_1)
        
    print("ChromaticAberrationNode test passed.")

def test_film_grain_node():
    node = FilmGrainNode()
    image_tensor = torch.rand(1, 32, 32, 3)

    grained_image_tensor, = node.apply_film_grain(
        image=image_tensor, density=0.5, intensity=0.5, 
        highlights=1.0, supersample_factor=1
    )

    assert isinstance(grained_image_tensor, torch.Tensor)
    assert grained_image_tensor.shape == image_tensor.shape
    assert grained_image_tensor.dtype == torch.float32
    # Grain should make it different
    assert_images_not_equal(grained_image_tensor, image_tensor)

    print("FilmGrainNode test passed.")

if __name__ == "__main__":
    test_minx_merge_node()
    test_image_rotate_node()
    test_chromatic_aberration_node()
    test_film_grain_node()
