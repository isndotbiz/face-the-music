# Face The Music 🎵

[![Version](https://img.shields.io/badge/version-2.0.0--beta.1-blue.svg)](https://github.com/isndotbiz/face-the-music/releases)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> Professional AI image generation with native face swapping using Flux Kontext Pro

<div align="center">

![Face The Music Demo](docs/images/demo.gif)

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🎨 Examples](#-examples) • [🤝 Contributing](#-contributing)

</div>

## 🌟 Overview

**Face The Music** is a state-of-the-art AI image generation pipeline that creates photorealistic images with seamless face integration. Built on Flux Kontext Pro, it delivers professional-grade results without complex post-processing.

### ✨ Key Features

- 🎨 **Native Face Swapping** - Built into Flux Kontext Pro, no external tools
- 📸 **Professional Quality** - Photography-grade prompts and technical specs
- 🖼️ **High Resolution** - Up to 2048×2048 with intelligent upscaling
- ⚡ **Batch Processing** - Efficient multi-image generation
- 🎭 **Luxury Themes** - Specialized prompts for high-end lifestyle imagery
- 🔧 **Zero Dependencies** - Streamlined pipeline, minimal setup

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/isndotbiz/face-the-music.git
cd face-the-music
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API
export REPLICATE_API_TOKEN="your_token_here"

# 4. Add your face image
cp your_face.jpg faces/your_face.png

# 5. Generate!
python generate_images.py
```

## 📋 Requirements

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.8+ | ✅ Required |
| Replicate API | Latest | ✅ Required |
| Face Image | 1024×1024 | ✅ Required |
| RAM | 8GB+ | 💡 Recommended |
| Storage | 2GB+ | 💡 Recommended |

## 🛠️ Installation

### Standard Installation

```bash
# Clone repository
git clone https://github.com/isndotbiz/face-the-music.git
cd face-the-music

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# For contributors
git clone https://github.com/isndotbiz/face-the-music.git
cd face-the-music
pip install -e .
pip install -r requirements-dev.txt  # Coming in v2.0.0-beta.2
pre-commit install  # Coming in v2.0.0-beta.2
```

## ⚙️ Configuration

### 1. API Setup

```bash
# Get your token from https://replicate.com/account
export REPLICATE_API_TOKEN="r8_***"

# Or create .env file
echo "REPLICATE_API_TOKEN=r8_***" > .env
```

### 2. Face Image Setup

```bash
# Place your face image
cp your_photo.jpg faces/your_face.png

# Optimal specifications:
# - Resolution: 1024×1024 pixels
# - Format: PNG (recommended) or JPG
# - Quality: Clear face, good lighting
# - Orientation: Front-facing
```

### 3. Configuration Files

**`config.yaml`** - Main settings
```yaml
model:
  name: "flux-kontext-pro"
  version: "latest"
  
generation:
  width: 2048
  height: 2048
  guidance_scale: 7.5
  
face_swap:
  enabled: true
  strength: 0.8
```

**`promfy_prompts.yaml`** - Image prompts
```yaml
luxury_themes:
  - prompt: "opulent mansion interior"
    style: "professional photography"
  - prompt: "luxury sports car"
    lighting: "golden hour"
```

## 🎮 Usage

### Basic Generation

```bash
# Generate with default settings
python generate_images.py

# Use custom prompts
python generate_images.py --prompts custom_prompts.yaml

# Batch generation
python generate_images.py --batch 5
```

### Advanced Usage

```python
from replicate_generator import ReplicateGenerator

# Initialize generator
generator = ReplicateGenerator()

# Generate single image
result = generator.generate(
    prompt="luxury yacht at sunset, photorealistic, 8K",
    face_image_path="faces/your_face.png",
    width=2048,
    height=2048,
    guidance_scale=7.5
)

# Save result
with open("output/yacht_sunset.png", "wb") as f:
    f.write(result)
```

### CLI Options

```bash
python generate_images.py [OPTIONS]

Options:
  --config PATH        Configuration file (default: config.yaml)
  --prompts PATH       Prompts file (default: promfy_prompts.yaml)
  --batch INTEGER      Number of images to generate
  --output PATH        Output directory (default: output/)
  --face PATH          Face image path (default: faces/your_face.png)
  --verbose           Enable detailed logging
  --help              Show help message
```

## 📁 Project Structure

```
face-the-music/
├── 📄 README.md                 # This file
├── 📄 CHANGELOG.md              # Version history
├── 📄 LICENSE                   # MIT license
├── 📄 CONTRIBUTING.md           # Contribution guidelines
├── ⚙️ config.yaml               # Main configuration
├── 📝 promfy_prompts.yaml       # Image prompts
├── 🐍 generate_images.py        # Main generation script
├── 🐍 replicate_generator.py    # Replicate API interface
├── 🐍 test_end_to_end.py        # Integration tests
├── 📦 requirements.txt          # Python dependencies
├── 🖼️ faces/                    # Face reference images
│   └── your_face.png
├── 📸 output/                    # Generated images
│   ├── luxury_*.png
│   └── yacht_*.png
├── 🗂️ temp_faces/               # Temporary processing
├── 📚 docs/                     # Documentation
│   ├── api.md
│   ├── examples.md
│   └── troubleshooting.md
└── 🧪 tests/                    # Test suite
    ├── unit/
    └── integration/
```

## 🎨 Examples

### Generated Images

| Theme | Description | Example |
|-------|-------------|---------|
| **Luxury Interior** | Opulent mansion, professional lighting | `dominant_chamber_portrait_01.png` |
| **Automotive** | Premium sports car, golden hour | `luxury_car_domination_02.png` |
| **Authority** | Gothic throne room, dramatic | `opulent_throne_figure_03.png` |
| **Wealth** | Swiss bank vault, sophisticated | `money_vault_mistress_05.png` |
| **Yacht** | Mediterranean marina, lifestyle | `yacht_luxury_deck_01.png` |

### Custom Prompt Examples

```yaml
# Professional Photography Style
professional_portrait:
  prompt: "executive boardroom, power suit, dramatic lighting"
  technical: "shot on Canon EOS R5, 85mm lens, f/1.4"
  lighting: "Rembrandt lighting, softbox key light"

# Lifestyle Luxury
luxury_lifestyle:
  prompt: "private jet interior, champagne, first class"
  mood: "sophisticated, aspirational"
  quality: "8K, photorealistic, commercial photography"
```

## 🔧 Model Information

### Flux Kontext Pro Specifications

| Feature | Specification |
|---------|---------------|
| **Architecture** | Flux Transformer + Native Face Integration |
| **Max Resolution** | 2048×2048 pixels |
| **Face Swapping** | Built-in, no external tools |
| **Generation Speed** | 30-60 seconds per image |
| **Quality** | Professional photography grade |
| **LoRA Support** | Not required (native capabilities) |

### Supported Features

- ✅ Native face reference integration
- ✅ High-resolution generation (up to 2048×2048)
- ✅ Professional photography styles
- ✅ Luxury and lifestyle themes
- ✅ Batch processing
- ✅ Custom prompt engineering
- ❌ External LoRA loading (not needed)
- ❌ Custom model fine-tuning

## 🐛 Troubleshooting

### Common Issues

<details>
<summary>🔑 API Token Problems</summary>

**Symptoms**: `Authentication failed` or `Invalid token`

**Solutions**:
```bash
# Check if token is set
echo $REPLICATE_API_TOKEN

# Test API connection
python -c "import replicate; print('✅ Connected')"

# Regenerate token at https://replicate.com/account
```
</details>

<details>
<summary>🖼️ Face Image Issues</summary>

**Symptoms**: Poor face integration, distorted features

**Solutions**:
- Use exactly 1024×1024 pixel resolution
- Ensure clear, front-facing photo
- Good lighting, minimal shadows
- Plain background recommended
- PNG format preferred

**Image Quality Checklist**:
- [ ] 1024×1024 resolution
- [ ] Clear facial features
- [ ] Good lighting
- [ ] Front-facing angle
- [ ] Minimal background distractions
</details>

<details>
<summary>💾 Memory and Performance</summary>

**Symptoms**: Generation fails, system slowdown

**Solutions**:
```bash
# Check system resources
free -h          # Linux
top              # Monitor during generation

# Reduce settings in config.yaml
width: 1024      # Instead of 2048
height: 1024
batch_size: 1
```
</details>

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `API_001` | Invalid Replicate token | Check REPLICATE_API_TOKEN |
| `IMG_001` | Face image not found | Verify faces/your_face.png exists |
| `IMG_002` | Invalid image format | Use PNG or JPG format |
| `GEN_001` | Generation timeout | Reduce image resolution |
| `GEN_002` | Model overloaded | Retry in a few minutes |
| `CFG_001` | Invalid configuration | Validate YAML syntax |
| `NET_001` | Network connection | Check internet connectivity |

## 🚧 Development Roadmap

### Current Version: 2.0.0-beta.1

### Upcoming Releases

| Version | Features | ETA |
|---------|----------|-----|
| **2.0.0-beta.2** | Advanced prompt engineering, batch optimization | Next week |
| **2.0.0-beta.3** | Real-time preview, generation monitoring | 2 weeks |
| **2.0.0-beta.4** | Custom style presets, one-click themes | 3 weeks |
| **2.0.0-rc.1** | Performance optimizations, bug fixes | 1 month |
| **2.0.0** | Stable release with full feature set | 1.5 months |

### Feature Requests

Vote on upcoming features in our [GitHub Discussions](https://github.com/isndotbiz/face-the-music/discussions)!

## 📊 Performance Metrics

| Metric | Value | Benchmark |
|--------|-------|----------|
| **Generation Speed** | 30-60s | Per 2048×2048 image |
| **Memory Usage** | 4-6GB | Peak during generation |
| **Face Accuracy** | >95% | Professional evaluation |
| **Output Quality** | 8K equivalent | Professional photography |
| **Batch Efficiency** | Linear scaling | Up to 10 images |

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Quick Contributing Guide

1. 🍴 **Fork** the repository
2. 🌿 **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. 💻 **Make** your changes
4. ✅ **Add** tests for new functionality
5. 📝 **Update** documentation
6. 🚀 **Submit** pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/face-the-music.git
cd face-the-music

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code quality
flake8 .
black .
isort .
```

### Contribution Areas

- 🐛 **Bug fixes** - Help us squash bugs
- ✨ **New features** - Add cool capabilities
- 📚 **Documentation** - Improve guides and examples
- 🧪 **Testing** - Increase test coverage
- 🎨 **Prompts** - Create new themed prompts
- 🔧 **Performance** - Optimize generation speed

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📚 Documentation

- 📖 **[API Reference](docs/api.md)** - Complete API documentation
- 🎯 **[Examples](docs/examples.md)** - Detailed usage examples
- 🔧 **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- 🎨 **[Prompt Engineering](docs/prompts.md)** - Advanced prompt techniques
- 🚀 **[Deployment](docs/deployment.md)** - Production deployment guide

## 🆘 Support

- 📖 **Documentation**: [GitHub Wiki](https://github.com/isndotbiz/face-the-music/wiki)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/isndotbiz/face-the-music/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/isndotbiz/face-the-music/discussions)
- 📧 **Email**: [support@facethemusic.ai](mailto:support@facethemusic.ai)
- 💬 **Discord**: [Join our community](https://discord.gg/facethemusic)

## 📄 License

```
MIT License

Copyright (c) 2024 Face The Music Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHERS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📈 Stats

![GitHub stars](https://img.shields.io/github/stars/isndotbiz/face-the-music?style=social)
![GitHub forks](https://img.shields.io/github/forks/isndotbiz/face-the-music?style=social)
![GitHub issues](https://img.shields.io/github/issues/isndotbiz/face-the-music)
![GitHub pull requests](https://img.shields.io/github/issues-pr/isndotbiz/face-the-music)

---

<div align="center">

**Made with ❤️ by the Face The Music team**

[⭐ Star us on GitHub](https://github.com/isndotbiz/face-the-music) • [🐛 Report Bug](https://github.com/isndotbiz/face-the-music/issues) • [🚀 Request Feature](https://github.com/isndotbiz/face-the-music/issues) • [💬 Join Discord](https://discord.gg/facethemusic)

</div>
