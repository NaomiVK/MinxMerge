# Comfy Minx Merge

Comfy Minx Merge is a custom node pack for ComfyUI that integrates multiple Language Model (LLM) services to generate creative prompts based on user inputted art styles and two images. The intent is for the LLM to merge everything into a new prompt.

## Features

- Supports LLM services: OpenAI, Anthropic (more to  come)
- Model selection dropdowns for each LLM service
- Editable system prompt for customized instructions
- Combines art style with elements from two input images
- Adjustable max tokens for output control

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

   Note: This package requires OpenAI API version 1.0.0 or higher. If you're upgrading from an older version, you may need to update your code to be compatible with the new API.

3. Set up your API keys as environment variables:
   - OPENAI_API_KEY
   - ANTHROPIC_API_KEY
   
## Compatibility

This package is compatible with:
- OpenAI API version 1.0.0 and above
- Anthropic API version 0.18.0 and above

If you encounter any issues related to API compatibility, please ensure you have the latest versions of the required libraries installed.

## Usage

1. Restart ComfyUI after installation.
2. Find the "Minx Merge" node in the node browser.
3. Connect two image inputs to the node.
4. Write your desired art style in the "art_style" field. Or convert widget to input and use it this way.
5. Choose the LLM service you want to use from the "llm_service" dropdown.
6. Select a model from the corresponding service's model dropdown (e.g., "openai_model" for OpenAI).
7. Adjust the max tokens and system prompt as needed.
8. Run the node to generate a creative prompt combining the art style and image elements.

## Customization

The default system prompt for the Comfy Minx Merge node is designed to create imaginative and cohesive scene descriptions by integrating elements from two images. Here's an overview of the default prompt:

```
Create imaginative prompts by integrating elements from two images into one cohesive and fully integrated scene. Ensure the description is concise, clear, and reflects the seamless fusion of both images. Users will specify an art style, and should be shortened to include the most important art style traits. The description of the images should avoid terms like 'blend' or 'merge,' and be capped at 120 words.

# Steps

1. Seamlessly integrate elements from both images into a single imaginative scene.
2. Avoid using terms related to merging or blending to describe the scene.
3. Ensure the final description is creative, coherent, and comprehensible within the 100 word limit.

# Output Format

- Begin with the user supplied art style.
- Follow with a unified scene description that reflects the integration of elements from both images.
- The entire output must be no longer than 120 words.
```

You can edit the `system_prompt` in the ComfyUI interface to customize the instructions given to the LLM. This allows you to fine-tune the output based on your specific needs, such as changing the word limit, adjusting the style of description, or focusing on particular aspects of the images or art style.

## Model Selection

Each LLM service (OpenAI, Anthropic) has its own model selection dropdown. These dropdowns are always visible, but you should only select a model for the service you're currently using. Select "none" for the services you're not using.

## Troubleshooting

If you encounter any issues with API keys or model selection, ensure that:
1. Your API keys are correctly set as environment variables.
2. You have an active subscription or access to the selected LLM service.
3. The selected model is available for your account/subscription level.
4. You've selected a model (not "none") for the LLM service you're using.


This script performs basic tests on the node's input types and model fetching capabilities.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
