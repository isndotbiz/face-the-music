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

### Installing InsightFace
For face swap functionality, you need to install InsightFace:

```bash
# Install InsightFace
pip install insightface

# For GPU support (recommended)
pip install onnxruntime-gpu

# For CPU-only (fallback)
pip install onnxruntime
```

**Note:** InsightFace requires specific system dependencies:
- **Linux/macOS**: Usually works out of the box
- **Windows**: May require Visual Studio Build Tools
- **GPU**: Requires CUDA-compatible GPU and drivers

### Downloading ONNX Models
InsightFace requires ONNX models for face detection and recognition:

```bash
# Create models directory
mkdir -p models/insightface

# Download the recommended models
# Option 1: Download via Python script
python -c "
import insightface
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
"

# Option 2: Manual download
# Download antelopev2.zip from:
# https://github.com/deepinsight/insightface/releases/tag/v0.7
# Extract to models/insightface/
```

**Available Models:**
- `antelopev2`: Recommended for best accuracy
- `buffalo_l`: Good balance of speed and accuracy
- `buffalo_m`: Faster, moderate accuracy
- `buffalo_s`: Fastest, basic accuracy

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

### Face Swap Backend Selection
You can choose between different face swap backends depending on your needs:

#### Using InsightFace (Recommended)
```bash
# Default - uses InsightFace if available
python generate_images.py --prompts promfy_prompts.yaml

# Explicitly specify InsightFace
python generate_images.py --prompts promfy_prompts.yaml --face-swap-backend insightface
```

#### Using Reactor (Alternative)
```bash
# Use Reactor backend
python generate_images.py --prompts promfy_prompts.yaml --face-swap-backend reactor
```

#### Configuration in YAML
You can also specify the backend in your prompt YAML file:

```yaml
default_settings:
  face_swap_backend: "insightface"  # or "reactor"
  
prompts:
  - text: "a portrait of a person"
    face_swap:
      backend: "insightface"  # Override per prompt
      source_face: "faces/your_face.png"
```

#### Backend Comparison
- **InsightFace**: Better accuracy, more stable, requires ONNX models
- **Reactor**: Faster setup, built into some ComfyUI installations
- **Auto-detection**: The application will automatically detect available backends

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