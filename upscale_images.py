import os
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import numpy as np

def maintain_aspect_ratio(width, height, target_height=1440):
    aspect_ratio = width / height
    new_width = int(aspect_ratio * target_height)
    return new_width, target_height

def process_image(input_path, output_path):
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image: {input_path}")
        return
        
    # Get current dimensions
    h, w = img.shape[:2]
    
    # Initialize model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    
    # Calculate target width while maintaining aspect ratio
    if h > 1440:
        target_width, target_height = maintain_aspect_ratio(w, h)
    else:
        target_width, target_height = w, h
    
    # Upscale
    output, _ = upsampler.enhance(img, outscale=4)
    
    # Resize to target dimensions if needed
    if output.shape[0] != target_height or output.shape[1] != target_width:
        output = cv2.resize(output, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Save the result
    cv2.imwrite(output_path, output)
    print(f"Processed: {input_path} -> {output_path}")

def main():
    input_dir = "input/images"
    output_dir = "temp/upscaled"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images in input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(input_path, output_path)

if __name__ == "__main__":
    main()
