# Face The Music: AI Image Generation with Native Face Swapping

A streamlined Python application for generating high-quality photorealistic images with native face swapping using Flux Kontext Pro.

## Features
- Batch photorealistic image generation from YAML prompts
- Native face swapping via Flux Kontext Pro (no post-processing needed)
- Professional photography-grade prompts with technical specifications
- High-resolution upscaling (2048×2048 output)
- CLI interface with progress tracking
- Luxury/dominance themed content optimization

## Requirements
- Python 3.8+
- Replicate API account and key
- High-quality face reference image (1024×1024 recommended)

## Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup
1. **Get Replicate API Key**:
   - Sign up at [replicate.com](https://replicate.com)
   - Generate an API token from your account settings

2. **Configure Environment**:
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env and add your Replicate API key
   REPLICATE_API_KEY=your_replicate_api_key_here
   ```

3. **Add Your Face Image**:
   - Place your reference face image in `faces/your_face.png`
   - **Recommended specs**: 1024×1024 pixels, high quality, clear face
   - **Format**: PNG or JPG

## Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Generate images using default prompts
python generate_images.py --prompts promfy_prompts.yaml
```

## How It Works
1. **Native Face Integration**: Flux Kontext Pro generates images with your face naturally integrated
2. **Professional Prompts**: Uses photography-grade technical specifications
3. **High-Resolution Output**: Automatically upscales to 2048×2048 pixels
4. **Luxury Themes**: Optimized for power/dominance/luxury content

## Example Output
The system generates 4 themed images:
- `dominant_chamber_portrait_01.png` - Antique chamber power scene
- `luxury_car_domination_02.png` - Premium automotive setting
- `opulent_throne_figure_03.png` - Gothic throne authority
- `money_vault_mistress_05.png` - Swiss bank vault wealth scene

## Custom Prompts
Edit `promfy_prompts.yaml` to customize:
- Prompts and scenarios
- Image dimensions
- Technical photography settings
- Face swap settings

## Output
- **Generated Images**: Saved to `output/` directory
- **Prepared Faces**: Cached in `output/temp_faces/`
- **Resolution**: 2048×2048 pixels (high-quality)
- **Format**: PNG with excellent detail

## Features
- ✅ **Native Face Swapping**: No post-processing required
- ✅ **Professional Quality**: Photography-grade prompts
- ✅ **High Resolution**: 2048×2048 output
- ✅ **Batch Processing**: Multiple images in one run
- ✅ **Luxury Themes**: Power/dominance/wealth scenarios
- ✅ **Clean Pipeline**: Streamlined, reliable workflow

## Troubleshooting
- **API Key Issues**: Ensure `REPLICATE_API_KEY` is set in `.env`
- **Face Quality**: Use high-resolution (1024×1024), clear face images
- **Generation Fails**: Check internet connection and Replicate API status
- **Low Quality**: Ensure face image is well-lit and high-resolution

## License
MIT
