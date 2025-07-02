#!/usr/bin/env python3
"""
Test end-to-end face swapping functionality without requiring ComfyUI server.
This simulates the complete workflow by creating mock generated images and then
applying face swapping.
"""

import os
import sys
import yaml
import typer
from dotenv import load_dotenv
from rich.progress import Progress
from PIL import Image, ImageDraw
from face_swapper import InsightFaceSwapper

app = typer.Typer()

def load_config(config_path: str = 'config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Allow environment variable overrides for face_swap settings
    if 'face_swap' in config:
        config['face_swap']['backend'] = os.getenv('FACE_SWAP_BACKEND', config['face_swap']['backend'])
        config['face_swap']['model_path'] = os.getenv('FACE_SWAP_MODEL_PATH', config['face_swap']['model_path'])
    
    return config

def load_prompts(prompts_path: str):
    with open(prompts_path, 'r') as f:
        return yaml.safe_load(f)

def create_mock_generated_image(name: str, width: int, height: int):
    """Create a mock generated image that simulates ComfyUI output."""
    # Use the source face as a base for our mock generated image
    source_face = Image.open('faces/your_face.png')
    if source_face.mode == 'RGBA':
        # Convert RGBA to RGB by creating a white background
        rgb_image = Image.new('RGB', source_face.size, (255, 255, 255))
        rgb_image.paste(source_face, mask=source_face.split()[-1])
        source_face = rgb_image
    
    # Resize and place on a background to simulate a generated scene
    mock_image = Image.new('RGB', (width, height), (50, 100, 150))  # Blue background
    
    # Add the face to the center of the image
    face_size = min(width, height) // 2
    resized_face = source_face.resize((face_size, face_size))
    
    # Paste the face in the center
    x = (width - face_size) // 2
    y = (height - face_size) // 2
    mock_image.paste(resized_face, (x, y))
    
    # Add some text to indicate this is a mock image
    draw = ImageDraw.Draw(mock_image)
    draw.text((10, 10), f"Mock {name}", fill=(255, 255, 255))
    
    return mock_image

@app.command()
def main(prompts: str = typer.Option('promfy_prompts.yaml', help="Prompt YAML file")):
    load_dotenv()
    config = load_config()
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    prompts_data = load_prompts(prompts)
    images = prompts_data.get('images', [])
    
    if not images:
        print("No images found in prompt file.")
        sys.exit(1)
    
    print("Testing end-to-end face swapping workflow...")
    print("(Using mock ComfyUI-generated images)")
    
    with Progress() as progress:
        task = progress.add_task("Processing images...", total=len(images))
        
        for name, img in images.items():
            prompt = img.get('prompt', '')
            sd_settings = config['sd_defaults'].copy()
            sd_settings.update(img.get('sd_settings', {}))
            face_swap_path = img.get('face_swap', {}).get('source')
            
            print(f"\\nProcessing: {name}")
            print(f"Prompt: {prompt}")
            print(f"Face swap: {'Yes' if face_swap_path else 'No'} ({'InsightFace' if face_swap_path else 'None'})")
            
            # Simulate ComfyUI image generation
            width = sd_settings.get('width', 1024)
            height = sd_settings.get('height', 1024)
            generated_image = create_mock_generated_image(name, width, height)
            
            # Save the "generated" image
            output_path = os.path.join(output_dir, f"{name}.png")
            generated_image.save(output_path, format='PNG')
            print(f"✓ Mock image generated: {output_path}")
            
            # Perform local face swap if configured
            if (face_swap_path and 'face_swap' in config and 
                config['face_swap'].get('backend') == 'insightface'):
                try:
                    print(f"🔄 Performing InsightFace face swap for {name}...")
                    
                    # Initialize the swapper with model path from config
                    swapper = InsightFaceSwapper(config['face_swap']['model_path'])
                    
                    # Perform improved face swap with 1024x1024 preparation
                    swapped_image = swapper.swap_faces_improved(
                        face_swap_path, 
                        generated_image, 
                        use_preparation=True, 
                        canvas_color='#FFFFFF'
                    )
                    
                    # Save the swapped image, overwriting the original
                    swapped_image.save(output_path, format='PNG')
                    print(f"✅ Face swap completed for {name}")
                    
                except Exception as e:
                    print(f"❌ Face swap failed for {name}: {e}")
                    # Continue with the original image if face swap fails
            else:
                print(f"⏭️  No face swap configured for {name}")
            
            progress.advance(task)
    
    print(f"\\n🎉 All images processed successfully!")
    print(f"📁 Check the '{output_dir}' directory for results.")
    
    # Summary
    print("\\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    for name, img in images.items():
        face_swap_path = img.get('face_swap', {}).get('source')
        output_path = os.path.join(output_dir, f"{name}.png")
        
        if os.path.exists(output_path):
            if face_swap_path:
                print(f"✅ {name}: Generated with InsightFace face swap")
            else:
                print(f"📸 {name}: Generated without face swap")
        else:
            print(f"❌ {name}: Failed to generate")

if __name__ == "__main__":
    app()
