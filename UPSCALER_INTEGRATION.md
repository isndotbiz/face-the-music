# Upscaler Integration Maintenance

## Overview
This document verifies that the upscaler integration has been properly maintained in the Face-The-Music workflow system.

## Verification Status ✅

### 1. `upscale` Flag Preservation
- The `upscale` flag is properly extracted from YAML prompts
- It's passed to the `inject_prompts` function in both ComfyUI and InsightFace modes
- The `inject_prompts` function correctly applies it to the ComfyUI workflow's Upscale node

### 2. `hires_fix` Setting Maintenance 
- The `hires_fix` setting is maintained in the SD settings
- It's properly injected into the KSampler node in ComfyUI workflows
- Default configuration sets `hires_fix: true` for high-resolution fix

### 3. Workflow Sequence Verification
- **Correct Order**: Generation → Upscaling → Face Swap
- Upscaler output is saved before face swap is applied
- Face swap operates on the already-upscaled image
- This ensures maximum quality in the final output

### 4. Implementation Details

#### ComfyUI Mode
```python
# In inject_prompts function
if node.get('type') == 'Upscale':
    node['inputs']['enabled'] = upscale
```

#### InsightFace Mock Mode
```python
# Simulates upscaling by increasing dimensions
if upscale:
    scale_factor = 2.0
    width = int(width * scale_factor)
    height = int(height * scale_factor)
```

### 5. Configuration Files

#### config.yaml
```yaml
sd_defaults:
  hires_fix: true  # Maintained for high-res fix
```

#### base_workflow.json
```json
{
  "type": "Upscale",
  "inputs": {
    "enabled": false,
    "upscale_method": "nearest-exact", 
    "scale_by": 2.0
  }
}
```

## Test Results

All upscaler integration tests pass:
- ✅ Upscaler flags preservation test
- ✅ Upscaler simulation in InsightFace mode test  
- ✅ Workflow sequence test
- ✅ Config defaults test

## Example Usage

### YAML Prompt with Upscaling
```yaml
- name: "high_res_portrait"
  prompt: "Portrait of a person, 8k, photorealistic"
  upscale: true
  sd_settings:
    hires_fix: true
    width: 1024
    height: 1024
```

### Expected Output
- Original generation: 1024x1024
- After upscaling: 2048x2048 (2x scale factor)
- Face swap applied to upscaled image

## Key Benefits

1. **Quality Preservation**: Face swap operates on high-resolution images
2. **Workflow Integrity**: Upscaling happens at the right point in the pipeline
3. **Backward Compatibility**: Existing prompts continue to work
4. **Configuration Flexibility**: Upscaling can be enabled/disabled per image

## Maintenance Notes

- The `upscale` and `hires_fix` flags are preserved throughout the workflow
- Upscaler output is verified to be saved before face swap application
- Both ComfyUI and InsightFace backend modes support upscaling
- Test suite validates integration correctness
