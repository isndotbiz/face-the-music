#!/usr/bin/env python3
"""
Test script to verify face swapping functionality using InsightFace.
This will test the face swapping on the generated placeholder images.
"""

import os
import yaml
from PIL import Image
from face_swapper import InsightFaceSwapper

def load_config(config_path: str = 'config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_face_swap():
    """Test face swapping functionality."""
    config = load_config()
    
    # Check if output images exist
    rockstar_path = 'output/rockstar_portrait.png'
    jazz_path = 'output/jazz_band.png'
    
    if not os.path.exists(rockstar_path):
        print(f"Warning: {rockstar_path} does not exist")
        return False
        
    if not os.path.exists(jazz_path):
        print(f"Warning: {jazz_path} does not exist")
        return False
    
    # Load face swap configuration
    face_swap_config = config.get('face_swap', {})
    model_path = face_swap_config.get('model_path', 'models/insightface/inswapper_128.onnx')
    source_face_path = 'faces/your_face.png'
    
    if not os.path.exists(model_path):
        print(f"Error: InsightFace model not found at {model_path}")
        return False
        
    if not os.path.exists(source_face_path):
        print(f"Error: Source face image not found at {source_face_path}")
        return False
    
    print("Testing InsightFace face swapping...")
    
    try:
        # Initialize the face swapper
        print(f"Loading InsightFace model from {model_path}")
        swapper = InsightFaceSwapper(model_path)
        print("✓ InsightFace model loaded successfully")
        
        # Test face swap on rockstar image
        print(f"Testing face swap on {rockstar_path}")
        rockstar_img = Image.open(rockstar_path)
        try:
            swapped_rockstar = swapper.swap_faces(source_face_path, rockstar_img)
            swapped_rockstar.save('output/rockstar_portrait_swapped.png')
            print("✓ Rockstar face swap completed - saved as rockstar_portrait_swapped.png")
        except Exception as e:
            print(f"✗ Rockstar face swap failed: {e}")
        
        # Test face swap on jazz image  
        print(f"Testing face swap on {jazz_path}")
        jazz_img = Image.open(jazz_path)
        try:
            swapped_jazz = swapper.swap_faces(source_face_path, jazz_img)
            swapped_jazz.save('output/jazz_band_swapped.png')
            print("✓ Jazz face swap completed - saved as jazz_band_swapped.png")
        except Exception as e:
            print(f"✗ Jazz face swap failed: {e}")
            
    except Exception as e:
        print(f"✗ Failed to initialize InsightFace swapper: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Face Swap Test")
    print("=" * 50)
    
    success = test_face_swap()
    
    print("\n" + "=" * 50)
    if success:
        print("Face swap test completed! Check output directory for results.")
    else:
        print("Face swap test encountered errors.")
    print("=" * 50)
