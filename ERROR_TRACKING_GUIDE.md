# Automated Error Tracking System

## Overview

The Face-The-Music project now includes a comprehensive automated error tracking system that:

- 🔍 **Automatically captures and logs errors** during pipeline execution
- 📝 **Adds detected errors to TODO.md** for project management
- ✅ **Tracks fixes and maintains a history** in FIXED_ERRORS.md
- 🎯 **Provides CLI tools** for error management
- 🚀 **Integrates seamlessly** with your existing Flux Kontext Pro pipeline

## Quick Start

### 1. Basic Usage
All major pipeline functions now have automatic error tracking. Just run your pipeline normally:

```bash
# Your pipeline will automatically track any errors
python3 optimal_pipeline.py input.jpg face.jpg --prompt "professional portrait"
```

### 2. Check Error Status
```bash
# View error summary
python3 error_tracker.py status

# List all active errors
python3 error_tracker.py list
```

### 3. Mark Errors as Fixed
```bash
# After fixing an error, mark it as resolved
python3 error_tracker.py fix <error_hash> -d "Description of fix"
```

## Files Created

The system creates and manages these files:

- **`errors_log.json`** - Raw error data and tracking information
- **`FIXED_ERRORS.md`** - Log of resolved issues with timestamps and descriptions
- **`TODO.md`** - Updated automatically with new error tasks

## Integration Points

### Flux Kontext Pro Pipeline

The following functions now have automatic error tracking:

```python
# In replicate_generator.py
@track_errors(context="flux_image_generation", severity="HIGH")
def generate_image(...)

@track_errors(context="image_upscaling", severity="MEDIUM") 
def upscale_image(...)

@track_errors(context="flux_workflow", severity="HIGH")
def generate_with_face_swap_and_upscale(...)

# In optimal_pipeline.py
@track_errors(context="stage1_face_swap", severity="HIGH")
def stage1_flux_kontext_generation(...)

@track_errors(context="stage2_resize", severity="MEDIUM")
def stage2_resize_or_skip(...)

@track_errors(context="stage3_sdxl_enhancement", severity="HIGH")
def stage3_sdxl_lora_photorealism(...)

@track_errors(context="stage4_postprocessing", severity="MEDIUM")
def stage4_professional_postprocessing(...)

@track_errors(context="optimal_workflow_complete", severity="HIGH")
def process_optimal_workflow(...)
```

## Usage Examples

### 1. Add Error Tracking to Your Own Functions

```python
from error_tracker import track_errors, error_tracking_context

# Method 1: Decorator
@track_errors(context="my_function", severity="HIGH")
def my_function():
    # Your code here
    pass

# Method 2: Context Manager
def my_other_function():
    with error_tracking_context("processing_block", "MEDIUM"):
        # Your code here
        pass
```

### 2. Using the CLI

```bash
# Check current status
python3 error_tracker.py status
# Output:
# 📊 Error Tracking Status
# ==============================
# Active Errors: 2
# Fixed Errors: 1
# Total Tracked: 3
# Last Updated: 2025-07-03T04:08:12

# List active errors
python3 error_tracker.py list
# Output:
# 📋 Active Errors:
# ====================
# 🔴 9ba5976d: FileNotFoundError
#    Message: Demo file not found error...
#    Severity: MEDIUM
#    Occurrences: 1

# Mark error as fixed
python3 error_tracker.py fix 9ba5976d -d "Fixed file path issue"
# Output:
# ✅ Marked error 9ba5976d as fixed
```

### 3. Running with Error Tracking

```bash
# Demo the system
python3 flux_kontext_with_tracking.py --demo

# Run Flux generation with tracking
python3 flux_kontext_with_tracking.py --flux-only --prompt "portrait" --face face.jpg

# Run full pipeline with tracking
python3 flux_kontext_with_tracking.py --input image.jpg --face face.jpg --prompt "portrait"
```

## Error Severity Levels

- **HIGH**: Critical errors that block workflow completion
- **MEDIUM**: Important errors that affect quality but allow continuation
- **LOW**: Minor issues or warnings

## How It Works

### 1. Error Detection
When an error occurs in a tracked function:
- Error details are captured (type, message, location, traceback)
- A unique hash is generated for deduplication
- Error is logged to `errors_log.json`

### 2. TODO Integration
For new errors:
- A task is automatically added to `TODO.md`
- Task includes error details and fix command
- Tasks are organized by severity level

### 3. Fix Tracking
When marking an error as fixed:
- Error is removed from `TODO.md`
- Details are moved to `FIXED_ERRORS.md` with timestamp
- Fix description is recorded

### 4. Deduplication
- Identical errors increment occurrence count
- Only one TODO item per unique error
- Last occurrence timestamp is updated

## File Structure

```
project/
├── error_tracker.py           # Main error tracking module
├── errors_log.json           # Raw error data
├── TODO.md                   # Updated with error tasks
├── FIXED_ERRORS.md          # Log of resolved issues
├── demo_error_tracking.py    # Demo script
├── flux_kontext_with_tracking.py  # Pipeline with tracking
└── ERROR_TRACKING_GUIDE.md  # This guide
```

## Configuration

### Error Tracker Initialization
```python
# Default configuration
tracker = ErrorTracker(
    errors_file="errors_log.json",
    todo_file="TODO.md", 
    fixed_file="FIXED_ERRORS.md"
)

# Custom configuration
tracker = ErrorTracker(
    errors_file="custom_errors.json",
    todo_file="custom_todo.md",
    fixed_file="custom_fixed.md"
)
```

### Decorator Options
```python
@track_errors(
    context="custom_context",    # Error location identifier
    severity="HIGH",             # HIGH, MEDIUM, or LOW
    auto_add_todo=True          # Auto-add to TODO list
)
```

## Workflow Integration

### Typical Workflow
1. **Development**: Write code with error tracking decorators
2. **Execution**: Run pipeline - errors are automatically captured
3. **Review**: Check `python3 error_tracker.py status` and `TODO.md`
4. **Fix**: Address issues listed in TODO
5. **Mark Fixed**: Use `python3 error_tracker.py fix <hash>`
6. **History**: Review `FIXED_ERRORS.md` for resolved issues

### Continuous Monitoring
- Errors accumulate over time with occurrence counts
- TODO list stays current with active issues
- Fixed issues maintain permanent record
- Quality metrics track error resolution rate

## Advanced Features

### Programmatic Access
```python
from error_tracker import _error_tracker

# Get error summary
summary = _error_tracker.get_error_summary()

# Get active errors
active_errors = _error_tracker.get_active_errors()

# Manual error tracking
error_hash = _error_tracker.track_error(
    error=exception_object,
    context="manual_tracking",
    severity="HIGH"
)

# Mark as fixed programmatically
_error_tracker.mark_error_fixed(error_hash, "Fixed manually")
```

### Custom TODO Sections
The system automatically finds and uses these TODO sections:
- `### HIGH PRIORITY` - for HIGH severity errors
- `### MEDIUM PRIORITY` - for MEDIUM severity errors  
- `### LOW PRIORITY` - for LOW severity errors
- Fallback: `## AUTO-TRACKED ERRORS` section

## Troubleshooting

### Common Issues

1. **Errors not appearing in TODO**
   - Check if `TODO.md` exists
   - Verify section headers exist
   - Check file permissions

2. **CLI commands not working**
   - Ensure you're in the project directory
   - Check Python path: `python3 error_tracker.py --help`
   - Verify error tracking files exist

3. **Duplicate errors**
   - This is expected behavior - occurrence count increases
   - Same error hash means identical error conditions

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Error tracker will show detailed logs
```

## Best Practices

1. **Use descriptive contexts** in decorators
2. **Set appropriate severity levels** based on impact
3. **Regularly review and fix** tracked errors
4. **Include meaningful fix descriptions** when marking resolved
5. **Keep TODO.md organized** with proper section headers
6. **Use the CLI regularly** to monitor error trends

## Examples in Your Project

The error tracking is now active in your key pipeline functions:

### Flux Generation
```python
# Automatically tracked
generator = ReplicateFluxGenerator()
image = generator.generate_image(
    prompt="professional portrait",
    model="flux-kontext-pro"
)
# Any errors are automatically captured and added to TODO
```

### Full Pipeline
```python
# Automatically tracked
pipeline = OptimalPipeline()
result = pipeline.process_optimal_workflow(
    input_image_path="input.jpg",
    face_image_path="face.jpg"
)
# Each stage tracks its own errors
```

This system ensures that no errors go unnoticed and provides a systematic approach to maintaining code quality and reliability in your Flux Kontext Pro pipeline.
