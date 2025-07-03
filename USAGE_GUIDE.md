# Face The Music Usage Guide 🎬

## Pipeline Comparison: Legacy vs Professional

### 🆕 Professional Pipeline v2.1 (Recommended)

**Best for**: Production work, high-quality outputs, professional projects

```bash
# 🧪 Test your system first
python test_professional_pipeline.py

# 📊 Check system readiness  
python pipeline_cli.py status

# 🎨 View available presets
python pipeline_cli.py presets

# 🚀 Process single image (recommended)
python pipeline_cli.py process \
  --input your_photo.jpg \
  --face faces/your_face.png \
  --preset balanced \
  --prompt "professional executive portrait"

# 🔄 Batch process directory
python pipeline_cli.py batch photos/ \
  --face faces/your_face.png \
  --preset cinema_grade \
  --max-count 10
```

**Features:**
- ✅ 4-stage professional processing
- ✅ Cinema-grade quality (4K-8K output)  
- ✅ Advanced upscaling with Real-ESRGAN
- ✅ SDXL refinement and post-processing
- ✅ Quality metrics and verification
- ✅ Professional CLI with presets
- ✅ Comprehensive error handling
- ✅ Rich progress display

### 🔄 Legacy Pipeline (Basic)

**Best for**: Quick testing, simple face swaps, learning

```bash
# Simple generation
python generate_images.py

# With custom prompts
python generate_images.py --prompts custom_prompts.yaml

# Batch generation
python generate_images.py --batch 5
```

**Features:**
- ✅ Fast face swapping with Flux Kontext Pro
- ✅ Basic upscaling to 2048×2048
- ✅ Luxury-themed prompts
- ✅ Batch processing
- ❌ No advanced post-processing
- ❌ Limited quality verification
- ❌ Basic CLI interface

## 🎯 When to Use Which Pipeline

### Use Professional Pipeline When:
- 🎬 Creating content for professional/commercial use
- 📸 Highest quality is required (8K, cinema-grade)
- 🔄 Processing multiple images efficiently  
- 🎨 Need advanced post-processing (color grading, noise reduction)
- 📊 Want quality metrics and verification
- ⚙️ Need preset configurations for different scenarios

### Use Legacy Pipeline When:
- 🚀 Quick testing and experimentation
- 💻 Limited system resources
- 📚 Learning the basics
- 🔄 Simple batch processing needs

## 🎨 Processing Presets Explained

### 🏆 Cinema Grade
- **Use case**: Professional film/video production
- **Quality**: Maximum (A+ grade)
- **Speed**: Slowest (2-5 minutes per image)
- **Features**: HDR processing, film grain, professional color grading
- **Output**: 8K resolution with cinema-quality post-processing

### ⭐ Ultra Quality  
- **Use case**: High-end photography, print work
- **Quality**: Ultra High (A grade)
- **Speed**: Slow (1-3 minutes per image)
- **Features**: Maximum detail preservation, advanced upscaling
- **Output**: 4K-8K resolution with enhanced details

### ✨ Balanced (Recommended)
- **Use case**: Most professional work, social media
- **Quality**: High (A- grade)
- **Speed**: Medium (30-90 seconds per image)
- **Features**: Good balance of quality and speed
- **Output**: 4K resolution with quality post-processing

### 🚀 Fast
- **Use case**: Testing, previews, quick iterations
- **Quality**: Standard (B+ grade)
- **Speed**: Fast (15-45 seconds per image)
- **Features**: Basic processing, quick results
- **Output**: 2K-4K resolution, minimal post-processing

## 📋 System Requirements by Preset

| Preset | RAM | GPU VRAM | Storage | Processing Time |
|--------|-----|-----------|---------|-----------------|
| Cinema Grade | 16GB+ | 8GB+ (optional) | 5GB+ | 2-5 min |
| Ultra Quality | 12GB+ | 6GB+ (optional) | 3GB+ | 1-3 min |
| Balanced | 8GB+ | 4GB+ (optional) | 2GB+ | 30-90s |
| Fast | 8GB+ | 2GB+ (optional) | 1GB+ | 15-45s |

## 🛠️ Quick Setup Commands

```bash
# 1. Install and setup
git clone https://github.com/isndotbiz/face-the-music.git
cd face-the-music
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API
export REPLICATE_API_TOKEN="your_token_here"

# 3. Test everything
python test_professional_pipeline.py

# 4. Process your first image
python pipeline_cli.py process \
  --input test_photo.jpg \
  --face faces/your_face.png \
  --preset balanced
```

## 🎯 Pro Tips

### 📸 Face Image Optimization
```bash
# Optimal face image specs:
# - Resolution: 1024×1024 pixels exactly
# - Format: PNG (preferred) or high-quality JPG
# - Lighting: Even, natural lighting
# - Background: Plain or removed
# - Orientation: Front-facing, eyes looking forward
# - Quality: No compression artifacts
```

### 🎨 Prompt Engineering
```bash
# Professional portrait example:
python pipeline_cli.py process \
  --input photo.jpg \
  --face faces/your_face.png \
  --preset cinema_grade \
  --prompt "executive boardroom, professional suit, dramatic lighting, shot on Canon EOS R5, 85mm lens"

# Lifestyle example:
python pipeline_cli.py process \
  --input photo.jpg \
  --face faces/your_face.png \
  --preset balanced \
  --prompt "luxury yacht deck, Mediterranean sunset, professional photography, golden hour"
```

### 🔄 Batch Processing Best Practices
```bash
# Process up to 10 images at once for stability
python pipeline_cli.py batch photos/ \
  --face faces/your_face.png \
  --preset balanced \
  --max-count 10 \
  --output-dir output/batch_results

# For large batches, process in chunks
for i in {1..5}; do
  python pipeline_cli.py batch photos/ \
    --face faces/your_face.png \
    --preset fast \
    --max-count 20 \
    --output-dir output/batch_$i
done
```

## 🔍 Quality Verification

The professional pipeline includes automatic quality assessment:

- **Facial Integrity Score**: How well facial features are preserved
- **Texture Preservation**: Detail retention and skin texture quality  
- **Color Accuracy**: Natural skin tones and color consistency
- **Artifact Detection**: Identification of unnatural elements
- **Overall Quality Grade**: A+ to C scale rating

## 📊 Output Structure

```
output/
├── stage1_face_swap/        # Initial face swap results
├── stage2_upscaled/         # Upscaled versions
├── stage3_refined/          # AI-refined outputs  
├── stage4_final/            # Final processed images
└── quality_reports/         # Quality assessment reports
```

## 🆘 Troubleshooting Quick Fixes

```bash
# System check
python pipeline_cli.py status

# Test pipeline components  
python test_professional_pipeline.py

# Clear cache and temp files
rm -rf temp/ output/temp_* __pycache__/

# Reset to known good state
git pull origin main
pip install -r requirements.txt --upgrade
```

## 📞 Getting Help

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/isndotbiz/face-the-music/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/isndotbiz/face-the-music/discussions)  
- 📚 **Documentation**: [README.md](README.md)
- 🧪 **Testing**: Run `python test_professional_pipeline.py`

---

*Happy generating! 🎨✨*
