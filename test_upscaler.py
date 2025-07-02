#!/usr/bin/env python3

import os
from PIL import Image
from dotenv import load_dotenv
from replicate_generator import ReplicateFluxGenerator

def test_upscaler():
    """Test the upscaler functionality with an existing image."""
    
    # Load environment variables
    load_dotenv()
    
    # Check if we have any generated images to test with
    test_images = [
        "output/dominant_chamber_portrait_01.png",
        "output/luxury_car_domination_02.png",
        "output/opulent_throne_figure_03.png"
    ]
    
    test_image_path = None
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image_path = img_path
            break
    
    if not test_image_path:
        print("❌ No test images found. Please generate some images first.")
        return False
    
    print(f"🧪 Testing upscaler with: {test_image_path}")
    
    try:
        # Load the test image
        test_image = Image.open(test_image_path)
        print(f"📏 Original image size: {test_image.size}")
        
        # Initialize the generator
        generator = ReplicateFluxGenerator()
        
        # Test upscaling
        upscaled = generator.upscale_image(
            image=test_image,
            scale=2,
            model='real-esrgan'
        )
        
        if upscaled and upscaled != test_image:
            # Save the upscaled test image
            output_path = "output/upscaler_test_result.png"
            upscaled.save(output_path)
            print(f"✅ Upscaler test successful!")
            print(f"📏 Upscaled image size: {upscaled.size}")
            print(f"💾 Saved to: {output_path}")
            return True
        else:
            print("❌ Upscaler test failed - no upscaled image returned")
            return False
            
    except Exception as e:
        print(f"❌ Upscaler test error: {e}")
        return False

if __name__ == "__main__":
    test_upscaler()
