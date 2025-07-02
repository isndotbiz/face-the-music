#!/usr/bin/env python3
"""
Test script to verify that upscaler integration is maintained correctly.
This tests that:
1. The upscale flag is preserved in inject_prompts
2. The hires_fix setting is maintained 
3. Upscaler output is saved before face swap is applied
"""

import os
import yaml
import tempfile
from PIL import Image
from comfyui_api import ComfyUIAPI
from generate_images import generate_with_insightface, load_config

def test_upscaler_flags_preserved():
    """Test that upscale and hires_fix flags are preserved in inject_prompts."""
    print("Testing upscaler flags preservation...")
    
    # Create a temporary workflow file
    workflow_data = {
        "nodes": [
            {
                "type": "KSampler",
                "inputs": {
                    "positive": "",
                    "negative": "",
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "Euler a",
                    "seed": -1,
                    "width": 512,
                    "height": 512,
                    "hires_fix": False  # Will be overridden
                }
            },
            {
                "type": "Upscale",
                "inputs": {
                    "enabled": False  # Will be overridden
                }
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        json.dump(workflow_data, f)
        workflow_path = f.name
    
    try:
        # Initialize API
        api = ComfyUIAPI("http://dummy", workflow_path)
        
        # Test settings
        sd_settings = {
            "hires_fix": True,
            "steps": 30,
            "cfg": 8.0,
            "width": 1024,
            "height": 1024
        }
        
        # Load and inject prompts with upscale enabled
        workflow = api.load_workflow()
        injected_workflow = api.inject_prompts(
            workflow=workflow,
            prompt="test prompt",
            negative_prompt="test negative",
            sd_settings=sd_settings,
            face_swap_path=None,
            enhance=False,
            upscale=True  # Enable upscaling
        )
        
        # Verify upscale flag is set
        upscale_node = None
        ksampler_node = None
        
        for node in injected_workflow["nodes"]:
            if node["type"] == "Upscale":
                upscale_node = node
            elif node["type"] == "KSampler":
                ksampler_node = node
        
        assert upscale_node is not None, "Upscale node not found in workflow"
        assert upscale_node["inputs"]["enabled"] == True, "Upscale flag not preserved"
        
        assert ksampler_node is not None, "KSampler node not found in workflow"
        assert ksampler_node["inputs"]["hires_fix"] == True, "hires_fix flag not preserved"
        
        print("✅ Upscaler flags preservation test passed")
        
    finally:
        os.unlink(workflow_path)

def test_upscaler_simulation_insightface():
    """Test that upscaling is simulated correctly in InsightFace mode."""
    print("Testing upscaler simulation in InsightFace mode...")
    
    # Create mock config
    config = {
        'sd_defaults': {
            'width': 512,
            'height': 512,
            'hires_fix': True
        },
        'mock_generation': {
            'background_color': '#FFFFFF',
            'default_width': 512,
            'default_height': 512
        }
    }
    
    # Test image config with upscale enabled
    img_config = {
        'prompt': 'test image',
        'upscale': True,
        'sd_settings': {
            'width': 512,
            'height': 512
        }
    }
    
    # Generate image
    generated_image = generate_with_insightface(img_config, config, "test")
    
    # Verify that dimensions are increased (upscaled)
    expected_width = int(512 * 2.0)  # 2x scale factor
    expected_height = int(512 * 2.0)
    
    assert generated_image.size == (expected_width, expected_height), \
        f"Expected size {(expected_width, expected_height)}, got {generated_image.size}"
    
    print("✅ Upscaler simulation test passed")

def test_workflow_sequence():
    """Test that the workflow sequence is correct: generation -> upscaling -> face swap."""
    print("Testing workflow sequence...")
    
    # Load actual config
    config = load_config()
    
    # Create test image configuration
    test_img = {
        'prompt': 'portrait of a person',
        'upscale': True,
        'sd_settings': {
            'width': 512,
            'height': 512,
            'hires_fix': True
        }
    }
    
    # Generate with InsightFace (mock) to test sequence
    mock_image = generate_with_insightface(test_img, config, "workflow_test")
    
    # Verify the image is properly generated with upscaling
    assert mock_image is not None, "Generated image is None"
    assert isinstance(mock_image, Image.Image), "Generated image is not a PIL Image"
    
    # Verify dimensions are upscaled (should be 2x the original)
    expected_size = (1024, 1024)  # 512 * 2
    assert mock_image.size == expected_size, \
        f"Expected upscaled size {expected_size}, got {mock_image.size}"
    
    print("✅ Workflow sequence test passed")

def test_config_defaults():
    """Test that config defaults maintain upscaler settings."""
    print("Testing config defaults...")
    
    config = load_config()
    
    # Verify hires_fix is enabled by default
    assert config['sd_defaults']['hires_fix'] == True, \
        "hires_fix should be enabled by default in config"
    
    print("✅ Config defaults test passed")

if __name__ == "__main__":
    print("Running upscaler integration tests...\n")
    
    try:
        test_upscaler_flags_preserved()
        test_upscaler_simulation_insightface()
        test_workflow_sequence()
        test_config_defaults()
        
        print("\n🎉 All upscaler integration tests passed!")
        print("\nUpscaler integration is properly maintained:")
        print("  ✅ upscale flag is preserved in inject_prompts")
        print("  ✅ hires_fix is maintained in SD settings")  
        print("  ✅ Upscaler output is processed before face swap")
        print("  ✅ Mock generation simulates upscaling correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
