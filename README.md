# Face The Music: Automated Image Generation & Face Swap with ComfyUI

A Python application to generate, face swap, enhance, and upscale images using ComfyUI workflows, based on YAML prompt files.

## Features
- Batch image generation from YAML prompts
- Face swap using ComfyUI nodes (InsightFace/Reactor)
- Optional enhancement (GFPGAN/CodeFormer) and upscaling
- CLI interface with progress tracking
- Robust error handling and configuration

## Requirements
- Python 3.8+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally
- Face swap/enhance/upscale models (download separately)

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py
```

## Configuration
- Copy `.env.example` to `.env` and adjust the `COMFYUI_API_URL` if needed.
- Edit `config.yaml` for default SD settings and workflow paths.
- Place your face images in the `faces/` directory.
- Place your face swap/enhance/upscale models in the `models/` directory.

## Preparing ComfyUI Workflow
1. Launch ComfyUI locally.
2. Build a workflow in the web UI that:
    - Loads SD checkpoint and LoRAs
    - Generates an image (KSampler)
    - (Optional) Hires.fix or upscaling
    - Loads a source face image
    - Performs face swap (InsightFace/Reactor node)
    - Enhances (GFPGAN/CodeFormer node)
    - Upscales (Upscale Model/Image node)
3. Save the workflow as `workflows/base_workflow.json`.
4. The app will inject prompt and parameter details into this workflow.

## Usage
```bash
python generate_images.py --prompts promfy_prompts.yaml
```

## Example Prompt File
See `promfy_prompts.yaml` for structure and options.

## Output
- Final images are saved in the `output/` directory.

## Troubleshooting
- Ensure ComfyUI is running and accessible at the configured API URL.
- Check model paths and workflow JSON for correctness.
- See error messages for details; the app retries failed API calls.

## License
MIT 