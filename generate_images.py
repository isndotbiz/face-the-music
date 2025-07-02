import os
import sys
import yaml
import typer
from dotenv import load_dotenv
from rich.progress import Progress
from comfyui_api import ComfyUIAPI
from PIL import Image, ImageDraw, ImageOps
# Face swapping now handled natively by Flux Kontext Pro
from replicate_generator import ReplicateFluxGenerator

app = typer.Typer()

# Define constant for temporary face directory
PREPARED_FACE_DIR = None  # Will be set in main() after output_dir is determined

def load_config(config_path: str = 'config.yaml', cli_backend: str = None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Allow environment variable overrides for face_swap settings
    if 'face_swap' in config:
        config['face_swap']['backend'] = os.getenv('FACE_SWAP_BACKEND', config['face_swap']['backend'])
        config['face_swap']['model_path'] = os.getenv('FACE_SWAP_MODEL_PATH', config['face_swap']['model_path'])

    # Override generation backend with CLI flag if provided
    if cli_backend is not None:
        config['generation']['backend'] = cli_backend

    return config

def load_prompts(prompts_path: str):
    with open(prompts_path, 'r') as f:
        return yaml.safe_load(f)

def save_image(image_data, output_path):
    # Ensure output path has .png extension
    if not output_path.lower().endswith('.png'):
        output_path = os.path.splitext(output_path)[0] + '.png'
    
    # If image_data is PIL Image, save as PNG
    if hasattr(image_data, 'save'):
        image_data.save(output_path, format='PNG')
    else:
        # If it's raw bytes, write directly (assuming it's already PNG format)
        with open(output_path, 'wb') as f:
            f.write(image_data)

def prepare_source_face_image(source_path: str, temp_dir: str) -> str:
    # Load image
    img = Image.open(source_path)
    
    # Convert to RGB with white background for better face detection
    if img.mode in ('RGBA', 'P'):
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        # Paste with transparency support
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Create square canvas (RGB with white background)
    size = 1024
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    
    # Center source on canvas with scaling
    img.thumbnail((size, size), Image.LANCZOS)
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    
    # Save prepared image
    filename = os.path.splitext(os.path.basename(source_path))[0] + "_prep.png"
    out_path = os.path.join(temp_dir, filename)
    canvas.save(out_path, format="PNG")
    return out_path

def generate_with_comfyui(api, img, config, name):
    """Generate image using ComfyUI workflow."""
    prompt = img.get('prompt', '')
    negative_prompt = img.get('negative_prompt', '')
    sd_settings = config['sd_defaults'].copy()
    sd_settings.update(img.get('sd_settings', {}))
    face_swap_path = img.get('face_swap', {}).get('source')
    enhance = img.get('enhance', False)
    upscale = img.get('upscale', False)
    
    workflow = api.load_workflow()
    workflow = api.inject_prompts(workflow, prompt, negative_prompt, sd_settings, face_swap_path, enhance, upscale)
    result = api.post_workflow(workflow)
    
    if not result:
        return None
    
    # Return the result for further processing
    return result

def generate_with_flux(img, config, name):
    """Generate image using Replicate Flux with face swap and upscaling."""
    try:
        # Initialize Flux generator
        flux_generator = ReplicateFluxGenerator()
        
        # Get configuration
        prompt = img.get('prompt', '')
        face_swap_path = img.get('face_swap', {}).get('source')
        enhance = img.get('enhance', False)
        upscale = img.get('upscale', False)
        
        # Merge SD settings
        sd_settings = config['sd_defaults'].copy()
        sd_settings.update(img.get('sd_settings', {}))
        
        # Use complete workflow: Flux → Face Swap → Upscale
        result_image = flux_generator.generate_with_face_swap_and_upscale(
            prompt=prompt,
            face_source_path=face_swap_path,
            config=config,
            sd_settings=sd_settings,
            enhance=enhance,
            upscale=upscale
        )
        
        return result_image
        
    except Exception as e:
        print(f"❌ Error in Flux generation for {name}: {e}")
        return None

def generate_with_insightface(img, config, name):
    """Generate mock image using InsightFace approach (mirrors test_end_to_end.py)."""
    sd_settings = config['sd_defaults'].copy()
    sd_settings.update(img.get('sd_settings', {}))
    
    # Get dimensions from SD settings or config defaults
    width = sd_settings.get('width', config.get('mock_generation', {}).get('default_width', 1024))
    height = sd_settings.get('height', config.get('mock_generation', {}).get('default_height', 1024))
    
    # Apply upscaling if enabled (mock behavior)
    upscale = img.get('upscale', False)
    if upscale:
        # Simulate upscaling by increasing dimensions
        scale_factor = 2.0
        width = int(width * scale_factor)
        height = int(height * scale_factor)
    
    # Get face swap source path
    face_swap_path = img['face_swap']['source']
    
    if face_swap_path and os.path.exists(face_swap_path):
        # Use the source face as a base for our mock generated image
        source_face = Image.open(face_swap_path)
        if source_face.mode == 'RGBA':
            # Convert RGBA to RGB by creating a white background
            rgb_image = Image.new('RGB', source_face.size, (255, 255, 255))
            rgb_image.paste(source_face, mask=source_face.split()[-1])
            source_face = rgb_image
    else:
        # Create a simple colored square if no face source is available
        source_face = Image.new('RGB', (256, 256), (200, 150, 100))  # Skin-like color
    
    # Get background color from config or use default
    bg_color_hex = config.get('mock_generation', {}).get('background_color', '#3264AA')
    # Convert hex to RGB tuple
    if bg_color_hex.startswith('#'):
        bg_color_hex = bg_color_hex[1:]
    bg_color = tuple(int(bg_color_hex[i:i+2], 16) for i in (0, 2, 4))
    
    # Create background
    mock_image = Image.new('RGB', (width, height), bg_color)
    
    # Add the face to the center of the image
    face_size = min(width, height) // 2
    resized_face = source_face.resize((face_size, face_size))
    
    # Paste the face in the center
    x = (width - face_size) // 2
    y = (height - face_size) // 2
    mock_image.paste(resized_face, (x, y))
    
    # Add optional text from prompt
    prompt = img.get('prompt', '')
    if prompt:
        draw = ImageDraw.Draw(mock_image)
        # Add prompt text at the top
        text_y = 10
        # Truncate prompt if too long
        display_text = prompt[:50] + '...' if len(prompt) > 50 else prompt
        draw.text((10, text_y), display_text, fill=(255, 255, 255))
    
    # Add identifier text to indicate this is a mock image
    draw = ImageDraw.Draw(mock_image)
    draw.text((10, height - 30), f"Mock {name}", fill=(255, 255, 255))
    
    return mock_image

@app.command()
def main(
    prompts: str = typer.Option('promfy_prompts.yaml', help="Prompt YAML file"),
    backend: str = typer.Option(None, help="Backend for generation to use")
):
    load_dotenv()
    config = load_config(cli_backend=backend)
    
    # Only initialize ComfyUI API if not using InsightFace backend
    api = None
    if config['generation']['backend'] != 'insightface':
        comfyui_api_url = os.getenv('COMFYUI_API_URL', config.get('comfyui_api_url', 'http://127.0.0.1:8188/prompt'))
        workflow_path = config.get('workflow_path', 'workflows/base_workflow.json')
        api = ComfyUIAPI(comfyui_api_url, workflow_path)
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the temporary face directory constant
    global PREPARED_FACE_DIR
    PREPARED_FACE_DIR = os.path.join(output_dir, "temp_faces")
    os.makedirs(PREPARED_FACE_DIR, exist_ok=True)
    
    # Clean up old files from previous runs
    if os.path.exists(PREPARED_FACE_DIR):
        for f in os.listdir(PREPARED_FACE_DIR):
            file_path = os.path.join(PREPARED_FACE_DIR, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    prompts_data = load_prompts(prompts)
    images = prompts_data.get('images', {})
    if not images:
        print("No images found in prompt file.")
        sys.exit(1)
    with Progress() as progress:
        task = progress.add_task("Generating images...", total=len(images))
        for name, image in images.items():
            img = {
                'prompt': image.get('prompt', ''),
                'negative_prompt': image.get('negative_prompt', ''),
                'sd_settings': image.get('sd_settings', {}),
                'face_swap': image.get('face_swap', {}),
                'enhance': image.get('enhance', False),
                'upscale': image.get('upscale', False)
            }
            
            # Prepare source face image if defined
            original_source = img['face_swap'].get('source')
            if original_source:
                prepared = prepare_source_face_image(original_source, PREPARED_FACE_DIR)
                img['face_swap']['source'] = prepared
            
            prompt = img.get('prompt', '')
            negative_prompt = img.get('negative_prompt', '')
            sd_settings = config['sd_defaults'].copy()
            sd_settings.update(img.get('sd_settings', {}))
            face_swap_path = img.get('face_swap', {}).get('source')
            enhance = img.get('enhance', False)
            upscale = img.get('upscale', False)
            # Choose generation backend
            backend = config['generation']['backend']
            
            if backend == 'flux':
                # Use Flux workflow with face swap and upscaling
                generated_image = generate_with_flux(img, config, name)
                if generated_image:
                    output_path = os.path.join(output_dir, f"{name}.png")
                    save_image(generated_image, output_path)
                    print(f"✅ Flux image with face swap and upscaling saved to: {output_path}")
                    result = {'image_saved': True}
                else:
                    print(f"❌ Failed to generate image with Flux: {name}")
                    progress.advance(task)
                    continue
            elif backend == 'comfyui':
                # Use ComfyUI workflow
                workflow = api.load_workflow()
                workflow = api.inject_prompts(workflow, prompt, negative_prompt, sd_settings, face_swap_path, enhance, upscale)
                result = api.post_workflow(workflow)
                if not result:
                    print(f"Failed to generate image: {name}")
                    progress.advance(task)
                    continue
            else:
                # Mock generation for InsightFace backend
                mock_image = generate_with_insightface(img, config, name)
                output_path = os.path.join(output_dir, f"{name}.png")
                mock_image.save(output_path, format='PNG')
                print(f"Upscaled mock image saved to: {output_path}")
                generated_image = mock_image
                result = {'image_saved': True}
            # Save output image (assume result contains image bytes or path)
            # This part may need to be adapted to your workflow's output format
            # IMPORTANT: The upscaler output (if enabled) is already applied at this point
            # since upscale=True is passed to inject_prompts and processed by ComfyUI
            output_path = os.path.join(output_dir, f"{name}.png")
            generated_image = None
            
            if 'image' in result:
                save_image(result['image'], output_path)
                # Verify the upscaled image is saved before face swap
                print(f"Upscaled image saved to: {output_path}")
                # Load the saved image for potential face swapping
                generated_image = Image.open(output_path)
            elif 'output_path' in result:
                # Copy file from output_path
                from shutil import copyfile
                copyfile(result['output_path'], output_path)
                # Verify the upscaled image is saved before face swap
                print(f"Upscaled image copied to: {output_path}")
                # Load the image for potential face swapping
                generated_image = Image.open(output_path)
            elif 'image_saved' in result and generated_image is not None:
                # Mock generation case - image already saved and loaded
                print(f"Mock upscaled image ready for face swap: {output_path}")
            else:
                print(f"No image data found for {name}")
                progress.advance(task)
                continue
            
            # Face swapping is now handled natively by Flux Kontext Pro during generation
            # No additional face swap processing needed
            progress.advance(task)
    print(f"\nAll images processed. Check the '{output_dir}' directory.")

if __name__ == "__main__":
    app() 