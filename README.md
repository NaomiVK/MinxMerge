# Comfy Minx Merge

Comfy Minx Merge is a comprehensive custom node pack for ComfyUI that provides advanced AI-powered image analysis, creative prompt generation, and high-quality image effects processing.

## Features

### AI & LLM Integration
- Vision-enabled LLM integration with OpenAI and Anthropic services  
- Multimodal AI models for intelligent image analysis and text generation
- Customizable instructions, examples, and prompts
- Flexible temperature and token controls with exponential backoff retry mechanism

### Image Processing & Effects
- Professional film grain effects with device selection (CPU/GPU)
- Chromatic aberration simulation for vintage/artistic looks
- Advanced object removal using LaMa (Large Mask Inpainting) model
- Flexible image rotation with multiple resampling methods

### Model Optimization
- LoRA-safe torch compilation for improved inference performance
- Multiple backend support (inductor, cudagraphs, nvfuser)
- Lazy compilation with thread safety
- Transformer-only compilation options

### Workflow Enhancement
- Seed management for reproducible results
- Professional image effects processing
- Device selection for optimal performance

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
   ```
   cd path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/comfy_minx_merge.git
   ```

2. Install the required dependencies:
   ```
   pip install -r comfy_minx_merge/requirements.txt
   ```

   Note: This package requires OpenAI API version 1.7.5 or higher. If you're upgrading from an older version, you may need to update your code to be compatible with the new API.

3. Set up your API keys as environment variables:
   - OPENAI_API_KEY
   - ANTHROPIC_API_KEY
   - GROQ_API_KEY

## Compatibility

This package is compatible with:
- OpenAI API version 1.7.5 and above (requires vision-enabled models like GPT-4 Vision)
- Anthropic API version 0.18.0 and above (requires vision-enabled models like Claude 3)
- ComfyUI's latest version

If you encounter any issues related to API compatibility, please ensure you have the latest versions of the required libraries installed.

## Usage

### Minx Merge Node

1. Restart ComfyUI after installation.
2. Find the "Minx Merge" node in the node browser.
3. Connect an image input to the node.
4. Choose the LLM service you want to use from the "llm_service" dropdown.
5. Select a model from the corresponding service's model dropdown (e.g., "openai_model" for OpenAI or "anthropic_model" for Anthropic).
6. Set your desired instruction, example, and prompt text.
7. Adjust max tokens and temperature as needed.
8. Set a seed value for reproducibility.
9. Toggle debug mode if you need detailed logging.
10. Run the node to process the image with the selected vision LLM.
11. Use the generated text output and seed for further operations in your workflow.

### Image Rotate Node

1. Find the "Image Rotate" node in the "Minx Merge/Image/Transform" category.
2. Connect an image input to the node.
3. Choose rotation mode:
   - "transpose" - uses PIL's transpose method with 90-degree increments
   - "internal" - uses PIL's rotate method with arbitrary angles
4. Set rotation angle (in degrees, should be multiples of 90)
5. Select resampling method (nearest, bilinear, or bicubic)
6. The output is the rotated image

### Chromatic Aberration Node

1. Find the "Chromatic Aberration" node in the "Minx Merge/Image/Filter" category.
2. Connect an image input to the node.
3. Adjust the RGB channel offset parameters:
   - `red_offset`: Horizontal offset for the red channel (positive = right, negative = left)
   - `green_offset`: Vertical offset for the green channel (positive = down, negative = up)
   - `blue_offset`: Vertical offset for the blue channel (positive = down, negative = up)
4. Set the `intensity` parameter to control the strength of the effect (0.0-1.0)
5. Adjust the `fade_radius` to control how the effect fades toward the edges of the image
6. The output is the image with chromatic aberration applied

### Film Grain Node

1. Find the "Film Grain" node in the "Minx Merge/Image/Filter" category.
2. Connect an image input to the node.
3. Adjust the grain parameters:
   - `density`: Controls how many noise pixels are added (0.01-1.0)
   - `intensity`: Controls the strength of the grain effect (0.01-1.0)
   - `highlights`: Adjusts the brightness of the final image (0.01-255.0)
   - `supersample_factor`: Controls the resolution at which the grain is generated (higher values = finer grain)
   - `device`: Choose execution device (AUTO, CPU, or GPU) - currently uses CPU regardless of setting
4. The output is the image with film grain applied

### LaMa Remove Object Node

1. Find the "LaMa Remove Object" node in the "MinxMerge/Image" category.
2. Connect an image and a mask input to the node.
3. Optional settings:
   - `device_mode`: Choose between AUTO (smart memory management), Prefer GPU, or CPU
4. The node automatically downloads the LaMa model on first use
5. The output is the image with masked objects intelligently removed

### Torch Compile LoRA Safe Node

1. Find the "Torch Compile LoRA Safe" node in the "Minx Merge/model/optimisation 🛠️" category.
2. Connect a MODEL input to the node.
3. Configure compilation settings:
   - `backend`: Choose between inductor, cudagraphs, or nvfuser
   - `mode`: Select default, reduce-overhead, or max-autotune
   - `fullgraph`: Enable for full graph compilation (more aggressive optimization)
   - `dynamic`: Enable dynamic shapes support
   - `compile_transformer_only`: Compile only transformer blocks instead of the entire UNet
4. The output is an optimized MODEL with improved inference performance
5. First inference will trigger compilation (expect initial delay)


## Customization

The Minx Merge node allows for extensive customization through three key text input fields:

### Instruction
Use this field to provide clear directions to the vision LLM about what you want it to do with the input image. For example:
```
Describe what you see in this image in detail. Focus on the main subjects, their appearances, the setting, and the overall mood.
```

### Example
This field lets you provide examples of the kind of output you're looking for, which helps guide the model to produce similar results:
```
Example output for a landscape image:
A serene mountain landscape at sunset. The peaks are dusted with snow, silhouetted against a gradient sky of orange, pink, and deep blue. Pine trees frame the foreground, while a still lake reflects the vibrant colors above.
```

### Prompt
This is where you can add specific questions or requests about the image:
```
What emotions does this image evoke? What story might it be telling?
```

By adjusting these three fields along with the temperature (higher for more creative responses, lower for more deterministic ones) and max tokens, you can precisely control how the vision LLM processes and responds to your images.

## Model Selection

The Minx Merge node supports two vision-enabled LLM services:

### OpenAI
Select from vision-capable models like:
- gpt-4-vision-preview
- gpt-4o

### Anthropic
Select from vision-capable models like:
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307

Each LLM service has its own model selection dropdown. These dropdowns are always visible, but you should only select a model for the service you're currently using. Select "none" for the service you're not using.

## Node Catalog

### Core AI Nodes
- **Minx Merge**: Vision-enabled LLM integration for image analysis

### Image Processing Nodes
- **Film Grain**: Professional film grain simulation with device selection
- **Chromatic Aberration**: RGB channel offset effects for vintage/artistic looks
- **LaMa Remove Object**: Advanced AI-powered object removal
- **Image Rotate**: Flexible image rotation with multiple resampling methods

### Model Optimization Nodes
- **Torch Compile LoRA Safe**: LoRA-compatible torch compilation for performance optimization

## Image Filter Effects

The image filter nodes provide several ways to add interesting visual effects to your images:

### Chromatic Aberration Effects
- Create subtle RGB channel shifts for a vintage camera look
- Simulate lens distortion and color fringing  
- Create sci-fi glitch aesthetics
- Add psychedelic color effects
- Highlight edges with colorful halos

### Film Grain Effects
- Add analog film texture to digital images
- Create vintage or retro photo looks
- Add atmospheric noise to flat images
- Simulate high-ISO photography
- Create artistic texture effects

### Object Removal
- Intelligent content-aware inpainting
- Seamless object removal with context preservation
- Support for complex backgrounds and textures
- Automatic model downloading and management

### Model Optimization
- LoRA-safe compilation for stable training workflows
- Performance improvements for inference speed
- Memory optimization through lazy compilation
- Support for various PyTorch backends
- Thread-safe compilation for concurrent workflows

## Troubleshooting

If you encounter any issues with API keys or model selection, ensure that:
1. Your API keys are correctly set as environment variables.
2. You have an active subscription or access to the selected LLM service.
3. The selected model is available for your account/subscription level.
4. You've selected a model (not "none") for the LLM service you're using.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.