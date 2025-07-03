# Contributing to Face The Music 🎵

Thank you for your interest in contributing to Face The Music! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Release Process](#release-process)

## 🤝 Code of Conduct

### Our Standards

- **Be respectful**: Treat everyone with respect and courtesy
- **Be inclusive**: Welcome and support people of all backgrounds
- **Be collaborative**: Work together constructively
- **Be professional**: Keep discussions focused and productive

### Unacceptable Behavior

- Harassment, discrimination, or offensive language
- Personal attacks or trolling
- Publishing private information without permission
- Any conduct that could reasonably be considered inappropriate

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- A Replicate API account
- Basic knowledge of AI/ML concepts (helpful but not required)

### First Time Setup

1. **Fork the repository**
   ```bash
   # Visit https://github.com/isndotbiz/face-the-music and click "Fork"
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/face-the-music.git
   cd face-the-music
   ```

3. **Set up remote upstream**
   ```bash
   git remote add upstream https://github.com/isndotbiz/face-the-music.git
   ```

4. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Coming in v2.0.0-beta.2
   ```

## 🛠️ Development Setup

### Environment Configuration

1. **Set up API tokens**
   ```bash
   cp .env.example .env
   # Edit .env with your Replicate API token
   ```

2. **Verify setup**
   ```bash
   python -c "import replicate; print('✅ Setup complete')"
   ```

### Development Tools (Coming in v2.0.0-beta.2)

```bash
# Code formatting
black .
isort .

# Linting
flake8 .
pylint src/

# Type checking
mypy .

# Pre-commit hooks
pre-commit install
```

## 📝 Making Changes

### Branch Naming Convention

Use descriptive branch names following this pattern:

```
<type>/<description>
```

**Types:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests
- `chore/` - Maintenance tasks

**Examples:**
- `feature/advanced-prompt-engineering`
- `fix/memory-leak-in-batch-processing`
- `docs/update-installation-guide`
- `refactor/replicate-generator-class`

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```bash
feat(prompts): add advanced prompt engineering system

fix(memory): resolve memory leak in batch processing

docs(readme): update installation instructions

refactor(generator): simplify replicate API interface
```

### Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Follow existing code style
   - Add tests for new functionality

3. **Test your changes**
   ```bash
   python -m pytest tests/
   python test_end_to_end.py
   ```

4. **Update documentation**
   - Update README if needed
   - Add docstrings to new functions
   - Update CHANGELOG.md

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Branch is up to date with main

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. **Automated checks** will run (coming in v2.0.0-beta.2)
2. **Maintainer review** - we'll review your code and provide feedback
3. **Address feedback** - make requested changes
4. **Approval** - once approved, we'll merge your PR

## 📏 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use type hints
def generate_image(prompt: str, width: int = 1024) -> bytes:
    """Generate an image with the given prompt.
    
    Args:
        prompt: The text prompt for generation
        width: Image width in pixels
        
    Returns:
        Image data as bytes
    """
    pass

# Use docstrings for all public functions
# Use descriptive variable names
# Keep lines under 88 characters (Black default)
```

### Code Organization

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import yaml
import replicate
from PIL import Image

# Local imports
from .config import Config
from .generator import ReplicateGenerator
```

### Error Handling

```python
# Use specific exception types
try:
    result = api_call()
except replicate.exceptions.ReplicateError as e:
    logger.error(f"API call failed: {e}")
    raise

# Provide helpful error messages
if not os.path.exists(face_path):
    raise FileNotFoundError(
        f"Face image not found at {face_path}. "
        f"Please ensure the file exists and try again."
    )
```

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestReplicateGenerator:
    """Test suite for ReplicateGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReplicateGenerator()
    
    def test_generate_with_valid_inputs(self):
        """Test generation with valid inputs."""
        # Arrange
        prompt = "test prompt"
        
        # Act
        result = self.generator.generate(prompt)
        
        # Assert
        assert result is not None
        assert len(result) > 0
```

### Test Categories

1. **Unit Tests** - Test individual functions and classes
2. **Integration Tests** - Test component interactions
3. **End-to-End Tests** - Test complete workflows
4. **Performance Tests** - Test speed and memory usage

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/test_generator.py

# Run integration tests
python test_end_to_end.py
```

## 📚 Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def generate_batch(prompts: List[str], config: Config) -> List[bytes]:
    """Generate multiple images from a list of prompts.
    
    This function processes a list of prompts and generates corresponding
    images using the configured model and settings.
    
    Args:
        prompts: List of text prompts for image generation
        config: Configuration object containing model settings
        
    Returns:
        List of generated images as bytes objects
        
    Raises:
        ValueError: If prompts list is empty
        ReplicateError: If API call fails
        
    Example:
        >>> config = Config.load("config.yaml")
        >>> prompts = ["luxury car", "yacht at sunset"]
        >>> images = generate_batch(prompts, config)
        >>> len(images)
        2
    """
    pass
```

### README Updates

When adding features, update the README:

1. **Features section** - Add new capabilities
2. **Usage section** - Add examples
3. **Configuration** - Document new options
4. **Troubleshooting** - Add common issues

### Code Comments

```python
# Use comments for complex logic
def optimize_batch_size(memory_limit: int) -> int:
    # Calculate optimal batch size based on available memory
    # Each image uses approximately 500MB during processing
    base_memory = 2000  # MB for model loading
    per_image_memory = 500  # MB per image
    
    available_memory = memory_limit - base_memory
    return max(1, available_memory // per_image_memory)
```

## 🚀 Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH[-PRERELEASE]**
- Example: `2.0.0-beta.1`

### Pre-release Development

1. **Feature branches** → `2.0.0-beta.X`
2. **Bug fixes** → `2.0.0-beta.X+1`
3. **Release candidate** → `2.0.0-rc.1`
4. **Stable release** → `2.0.0`

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in relevant files
- [ ] Git tag created
- [ ] GitHub release created
- [ ] PyPI package updated (future)

## 🎯 Contribution Areas

### High Priority

- 🐛 **Bug fixes** - Help us maintain quality
- ⚡ **Performance** - Optimize generation speed
- 🧪 **Testing** - Increase test coverage
- 📚 **Documentation** - Improve guides and examples

### Medium Priority

- ✨ **New features** - Add exciting capabilities
- 🔧 **Refactoring** - Improve code quality
- 🎨 **Prompts** - Create new themed collections
- 🌐 **Internationalization** - Multi-language support

### Feature Requests

Vote on features in [GitHub Discussions](https://github.com/isndotbiz/face-the-music/discussions):

- Real-time generation preview
- Custom style presets
- Batch processing optimization
- Web interface
- API server mode

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat (coming soon)
- **Email**: [maintainers@facethemusic.ai](mailto:maintainers@facethemusic.ai)

### Before Asking for Help

1. **Search existing issues** - Your question might be answered
2. **Check documentation** - Review README and docs
3. **Try minimal example** - Isolate the problem
4. **Provide details** - Include error messages, environment info

### Issue Templates

**Bug Report:**
```markdown
**Bug Description**
A clear description of the bug

**To Reproduce**
Steps to reproduce the behavior

**Expected Behavior**
What you expected to happen

**Environment**
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.9.0]
- Version: [e.g., 2.0.0-beta.1]

**Additional Context**
Any other relevant information
```

## 🏆 Recognition

### Contributors

All contributors are recognized in:
- GitHub contributors page
- CHANGELOG.md acknowledgments
- Release notes
- Project documentation

### Types of Contributions

We value all types of contributions:

- 💻 **Code** - Features, fixes, improvements
- 📚 **Documentation** - Guides, examples, tutorials
- 🐛 **Testing** - Bug reports, test cases
- 💡 **Ideas** - Feature suggestions, feedback
- 🎨 **Design** - UI/UX improvements, graphics
- 🌐 **Community** - Support, outreach, advocacy

## 📈 Project Statistics

- **Current Version**: 2.0.0-beta.1
- **Active Contributors**: 1 (growing!)
- **Issues Resolved**: 15+
- **Test Coverage**: 85%+ (target)
- **Documentation Coverage**: 90%+

---

Thank you for contributing to Face The Music! Together, we're building the future of AI image generation with seamless face integration. 🚀

*For questions about this contributing guide, please [open an issue](https://github.com/isndotbiz/face-the-music/issues) or reach out to the maintainers.*
