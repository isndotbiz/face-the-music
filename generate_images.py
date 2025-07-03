#!/usr/bin/env python3
"""
Face The Music - Clean Image Generation Script
Using only Flux Kontext Pro with native face swapping

Version: 2.1-PROFESSIONAL  
Author: Face The Music Team
"""

import os
import sys
import yaml
import typer
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from PIL import Image, ImageDraw, ImageOps
from replicate_generator import ReplicateFluxGenerator
from pathlib import Path
import time

console = Console()
app = typer.Typer()

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        console.print(f"[red]❌ Config file not found: {config_path}[/red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]❌ Invalid YAML in config: {e}[/red]")
        sys.exit(1)

def load_prompts(prompts_path: str) -> dict:
    """Load prompts from YAML file"""
    try:
        with open(prompts_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]❌ Prompts file not found: {prompts_path}[/red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]❌ Invalid YAML in prompts: {e}[/red]")
        sys.exit(1)

def prepare_face_image(source_path: str, temp_dir: str) -> str:
    """Prepare source face image for optimal face swapping"""
    if not os.path.exists(source_path):
        console.print(f"[red]❌ Face image not found: {source_path}[/red]")
        return None
        
    try:
        # Load and process image
        img = Image.open(source_path)
        
        # Convert to RGB with white background
        if img.mode in ('RGBA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Create 1024x1024 canvas for optimal face detection
        size = 1024
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        
        # Center and scale image
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y))
        
        # Save prepared image
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"{Path(source_path).stem}_1024_prepared.png"
        out_path = os.path.join(temp_dir, filename)
        canvas.save(out_path, format="PNG")
        
        console.print(f"[green]✅ Face image prepared: {out_path}[/green]")
        return out_path
        
    except Exception as e:
        console.print(f"[red]❌ Error preparing face image: {e}[/red]")
        return None

def generate_with_flux(image_config: dict, config: dict, name: str, flux_generator: ReplicateFluxGenerator) -> Image.Image:
    """Generate image using Flux Kontext Pro with native face swapping"""
    try:
        console.print(f"[cyan]🎨 Generating {name} with Flux Kontext Pro...[/cyan]")
        
        # Get configuration
        prompt = image_config.get('prompt', '')
        face_swap_config = image_config.get('face_swap', {})
        face_source_path = face_swap_config.get('source')
        upscale = image_config.get('upscale', False)
        
        # Merge SD settings
        sd_settings = config.get('sd_defaults', {})
        sd_settings.update(image_config.get('sd_settings', {}))
        
        # Use complete Flux workflow with face swapping
        result_image = flux_generator.generate_with_face_swap_and_upscale(
            prompt=prompt,
            face_source_path=face_source_path,
            config=config,
            sd_settings=sd_settings,
            enhance=image_config.get('enhance', False),
            upscale=upscale
        )
        
        if result_image:
            console.print(f"[green]✅ Generated {name} successfully![/green]")
        else:
            console.print(f"[red]❌ Failed to generate {name}[/red]")
            
        return result_image
        
    except Exception as e:
        console.print(f"[red]❌ Error generating {name}: {e}[/red]")
        return None

def create_fallback_image(image_config: dict, config: dict, name: str) -> Image.Image:
    """Create a fallback mock image when generation fails"""
    try:
        # Get dimensions
        sd_settings = config.get('sd_defaults', {})
        sd_settings.update(image_config.get('sd_settings', {}))
        
        width = sd_settings.get('width', 1024)
        height = sd_settings.get('height', 1024)
        
        # Apply upscaling if enabled
        if image_config.get('upscale', False):
            width *= 2
            height *= 2
        
        # Create background
        bg_color = (50, 100, 150)  # Nice blue
        mock_image = Image.new('RGB', (width, height), bg_color)
        
        # Add face if available
        face_swap_config = image_config.get('face_swap', {})
        face_source_path = face_swap_config.get('source')
        
        if face_source_path and os.path.exists(face_source_path):
            face_img = Image.open(face_source_path)
            face_size = min(width, height) // 3
            face_img = face_img.resize((face_size, face_size), Image.Resampling.LANCZOS)
            
            # Center the face
            x = (width - face_size) // 2
            y = (height - face_size) // 2
            mock_image.paste(face_img, (x, y))
        
        # Add text overlay
        draw = ImageDraw.Draw(mock_image)
        prompt = image_config.get('prompt', '')[:50]
        draw.text((10, 10), f"Fallback: {name}", fill=(255, 255, 255))
        draw.text((10, 30), prompt, fill=(255, 255, 255))
        
        return mock_image
        
    except Exception as e:
        console.print(f"[red]❌ Error creating fallback image: {e}[/red]")
        return None

@app.command()
def main(
    prompts: str = typer.Option('promfy_prompts.yaml', help="Path to prompts YAML file"),
    config_file: str = typer.Option('config.yaml', help="Path to configuration file"),
    output_dir: str = typer.Option(None, help="Output directory (overrides config)"),
    batch_size: int = typer.Option(None, help="Number of images to generate"),
    face_image: str = typer.Option(None, help="Face reference image path"),
    verbose: bool = typer.Option(False, help="Enable verbose output")
):
    """
    Generate images using Flux Kontext Pro with native face swapping
    
    This script processes prompts from a YAML file and generates high-quality
    images with seamless face integration using Flux Kontext Pro.
    """
    
    # Load environment variables
    load_dotenv()
    
    # Display header
    console.print(Panel.fit(
        "🎵 [bold blue]Face The Music - Image Generation[/bold blue]\\n"
        "Professional face swapping with Flux Kontext Pro",
        title="Face The Music v2.1"
    ))
    
    # Load configuration
    config = load_config(config_file)
    
    # Override output directory if specified
    if output_dir:
        config['output_dir'] = output_dir
    
    output_path = Path(config.get('output_dir', 'output'))
    output_path.mkdir(exist_ok=True)
    
    # Setup temp directory for face processing
    temp_dir = output_path / "temp_faces"
    temp_dir.mkdir(exist_ok=True)
    
    # Load prompts
    prompts_data = load_prompts(prompts)
    images = prompts_data.get('images', {})
    
    if not images:
        console.print("[red]❌ No images found in prompt file[/red]")
        sys.exit(1)
    
    # Limit batch size if specified
    if batch_size:
        images = dict(list(images.items())[:batch_size])
    
    console.print(f"[blue]📝 Found {len(images)} images to generate[/blue]")
    
    # Check API token
    if not os.getenv('REPLICATE_API_TOKEN'):
        console.print("[red]❌ REPLICATE_API_TOKEN not set![/red]")
        console.print("Set it with: export REPLICATE_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    try:
        # Initialize Flux generator
        console.print("[blue]🔧 Initializing Flux Kontext Pro...[/blue]")
        flux_generator = ReplicateFluxGenerator()
        
        # Process each image
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Generating images...", total=len(images))
            
            for name, image_config in images.items():
                progress.update(task, description=f"Processing {name}")
                
                # Override face image if specified via CLI
                if face_image:
                    if 'face_swap' not in image_config:
                        image_config['face_swap'] = {}
                    image_config['face_swap']['source'] = face_image
                
                # Prepare face image if specified
                face_swap_config = image_config.get('face_swap', {})
                original_face_path = face_swap_config.get('source')
                
                if original_face_path:
                    prepared_face_path = prepare_face_image(original_face_path, str(temp_dir))
                    if prepared_face_path:
                        image_config['face_swap']['source'] = prepared_face_path
                
                # Generate image
                start_time = time.time()
                generated_image = generate_with_flux(image_config, config, name, flux_generator)
                generation_time = time.time() - start_time
                
                # Fallback to mock image if generation failed
                if not generated_image:
                    console.print(f"[yellow]⚠️  Creating fallback image for {name}[/yellow]")
                    generated_image = create_fallback_image(image_config, config, name)
                
                # Save the result
                if generated_image:
                    output_file = output_path / f"{name}.png"
                    generated_image.save(output_file, format='PNG')
                    
                    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                    console.print(
                        f"[green]✅ Saved {name}: {output_file} "
                        f"({generated_image.size[0]}×{generated_image.size[1]}, "
                        f"{file_size:.1f}MB, {generation_time:.1f}s)[/green]"
                    )
                else:
                    console.print(f"[red]❌ Failed to generate {name}[/red]")
                
                progress.advance(task)
        
        console.print(f"\\n[bold green]🎉 Generation complete![/bold green]")
        console.print(f"📁 Output directory: {output_path}")
        console.print(f"📊 Generated {len(images)} images")
        
    except KeyboardInterrupt:
        console.print("\\n[yellow]⚠️  Generation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\\n[red]❌ Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    app()
