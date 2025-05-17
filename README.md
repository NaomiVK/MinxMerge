# Comfy Minx Merge

Comfy Minx Merge is a custom node pack for ComfyUI that integrates multiple Language Model (LLM) services to process images and generate text, along with utility nodes for image manipulation.

## Features

- Vision-enabled LLM integration with OpenAI and Anthropic services
- Process images with multimodal AI models to generate text output
- Customizable instructions, examples, and prompts
- Flexible input options with adjustable temperature and max tokens
- Exponential backoff retry mechanism for API failures
- Image Rotate node: Rotate images with various methods and sampling options

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

## Troubleshooting

If you encounter any issues with API keys or model selection, ensure that:
1. Your API keys are correctly set as environment variables.
2. You have an active subscription or access to the selected LLM service.
3. The selected model is available for your account/subscription level.
4. You've selected a model (not "none") for the LLM service you're using.

## Testing

To verify the functionality of the Comfy Minx Merge node pack, you can run the included test script:

```
python comfy_minx_merge/test_minx_merge.py
```

This script performs basic tests on the node's input types and model fetching capabilities.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
