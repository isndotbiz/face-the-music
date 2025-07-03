#!/usr/bin/env python3
"""
Test end-to-end face swapping functionality using Flux Kontext Pro.
This tests the complete workflow using the working Flux generator.
"""

import os
import sys
import yaml
import typer
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from PIL import Image, ImageDraw
from replicate_generator import ReplicateFluxGenerator
import time

console = Console()
app = typer.Typer()

def load_config(config_path: str = 'config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]❌ Config file not found: {config_path}[/red]")
        sys.exit(1)

def load_prompts(prompts_path: str):
    """Load prompts from YAML file"""
    try:
        with open(prompts_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]❌ Prompts file not found: {prompts_path}[/red]")
        sys.exit(1)

def test_flux_generation(name: str, image_config: dict, flux_generator: ReplicateFluxGenerator) -> Image.Image:
    """Test actual Flux Kontext Pro generation with face swapping"""
    try:
        console.print(f"[cyan]🎨 Testing Flux generation for {name}[/cyan]")
        
        prompt = image_config.get('prompt', '')
        face_swap_config = image_config.get('face_swap', {})
        face_path = face_swap_config.get('source')
        
        # Generate with Flux Kontext Pro
        result_image = flux_generator.generate_image(
            prompt=prompt,
            width=1024,
            height=1024,
            model='flux-kontext-pro',
            reference_image_path=face_path,
            steps=25,
            guidance_scale=3.5
        )
        
        if result_image:
            console.print(f"[green]✅ Flux generation successful for {name}[/green]")
        else:
            console.print(f"[red]❌ Flux generation failed for {name}[/red]")
            
        return result_image
        
    except Exception as e:
        console.print(f"[red]❌ Error in Flux generation: {e}[/red]")
        return None

@app.command()
def main(
    prompts: str = typer.Option('promfy_prompts.yaml', help="Path to prompts YAML file"),
    config_file: str = typer.Option('config.yaml', help="Path to configuration file"),
    max_images: int = typer.Option(3, help="Maximum number of images to test"),
    verbose: bool = typer.Option(False, help="Enable verbose output")
):
    """
    Test end-to-end face swapping functionality using Flux Kontext Pro
    
    This script tests the complete workflow by generating actual images
    with face swapping using the professional Flux Kontext Pro pipeline.
    """
    
    # Load environment variables
    load_dotenv()
    
    # Display header
    console.print(Panel.fit(
        "🧪 [bold blue]End-to-End Pipeline Test[/bold blue]\\n"
        "Testing Flux Kontext Pro with native face swapping",
        title="Face The Music - Pipeline Test"
    ))
    
    # Load configuration and prompts
    config = load_config(config_file)
    prompts_data = load_prompts(prompts)
    images = prompts_data.get('images', {})
    
    if not images:
        console.print("[red]❌ No images found in prompt file[/red]")
        sys.exit(1)
    
    # Limit number of test images
    if max_images:
        images = dict(list(images.items())[:max_images])
    
    console.print(f"[blue]🧪 Testing {len(images)} images[/blue]")
    
    # Setup output directory
    output_dir = config.get('output_dir', 'output')
    test_output_dir = os.path.join(output_dir, 'test_end_to_end')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Check API token
    if not os.getenv('REPLICATE_API_TOKEN'):
        console.print("[red]❌ REPLICATE_API_TOKEN not set![/red]")
        console.print("Set it with: export REPLICATE_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Check face image
    face_path = None
    for name, img_config in images.items():
        face_swap_config = img_config.get('face_swap', {})
        if face_swap_config.get('source'):
            face_path = face_swap_config['source']
            break
    
    if face_path and not os.path.exists(face_path):
        console.print(f"[red]❌ Face reference image not found: {face_path}[/red]")
        sys.exit(1)
    
    try:
        # Initialize Flux generator
        console.print("[blue]🔧 Initializing Flux Kontext Pro...[/blue]")
        flux_generator = ReplicateFluxGenerator()
        
        # Test results tracking
        results = []
        total_time = 0
        
        # Process each image
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Testing pipeline...", total=len(images))
            
            for name, image_config in images.items():
                progress.update(task, description=f"Testing {name}")
                
                start_time = time.time()
                
                # Test generation
                generated_image = test_flux_generation(name, image_config, flux_generator)
                
                generation_time = time.time() - start_time
                total_time += generation_time
                
                # Save result
                if generated_image:
                    output_file = os.path.join(test_output_dir, f"test_{name}.png")
                    generated_image.save(output_file, format='PNG')
                    
                    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                    
                    result = {
                        'name': name,
                        'status': 'success',
                        'output_file': output_file,
                        'size': generated_image.size,
                        'file_size_mb': file_size,
                        'generation_time': generation_time,
                        'prompt': image_config.get('prompt', '')[:50] + '...',
                        'face_swap': bool(image_config.get('face_swap', {}).get('source'))
                    }
                    
                    console.print(f"[green]✅ {name}: Generated successfully ({generation_time:.1f}s)[/green]")
                    
                else:
                    result = {
                        'name': name,
                        'status': 'failed',
                        'generation_time': generation_time,
                        'prompt': image_config.get('prompt', '')[:50] + '...',
                        'face_swap': bool(image_config.get('face_swap', {}).get('source'))
                    }
                    
                    console.print(f"[red]❌ {name}: Generation failed ({generation_time:.1f}s)[/red]")
                
                results.append(result)
                progress.advance(task)
        
        # Display test results summary
        _display_test_summary(results, total_time, test_output_dir)
        
    except KeyboardInterrupt:
        console.print("\\n[yellow]⚠️  Testing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\\n[red]❌ Testing error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def _display_test_summary(results: list, total_time: float, output_dir: str):
    """Display comprehensive test results summary"""
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    console.print("\\n" + "="*80)
    console.print(Panel.fit(
        f"🧪 [bold blue]End-to-End Test Results[/bold blue]\\n"
        f"✅ Successful: {len(successful)}/{len(results)}\\n"
        f"❌ Failed: {len(failed)}/{len(results)}\\n"
        f"⏱️  Total Time: {total_time:.1f}s\\n"
        f"📁 Output: {output_dir}",
        title="Test Summary"
    ))
    
    # Detailed results table
    if successful:
        console.print("\\n[bold green]✅ Successful Generations:[/bold green]")
        for result in successful:
            console.print(
                f"  📸 {result['name']}: {result['size'][0]}×{result['size'][1]} "
                f"({result['file_size_mb']:.1f}MB, {result['generation_time']:.1f}s)"
            )
            if result['face_swap']:
                console.print(f"     👤 Face swapping: Enabled")
            console.print(f"     📝 Prompt: {result['prompt']}")
    
    if failed:
        console.print("\\n[bold red]❌ Failed Generations:[/bold red]")
        for result in failed:
            console.print(f"  ❌ {result['name']}: Failed after {result['generation_time']:.1f}s")
            console.print(f"     📝 Prompt: {result['prompt']}")
    
    # Performance metrics
    if successful:
        avg_time = sum(r['generation_time'] for r in successful) / len(successful)
        avg_size = sum(r['file_size_mb'] for r in successful) / len(successful)
        
        console.print("\\n[bold blue]📊 Performance Metrics:[/bold blue]")
        console.print(f"  ⏱️  Average generation time: {avg_time:.1f}s")
        console.print(f"  📏 Average file size: {avg_size:.1f}MB")
        console.print(f"  🎯 Success rate: {len(successful)/len(results)*100:.1f}%")
    
    # Overall status
    if len(successful) == len(results):
        console.print("\\n[bold green]🎉 All tests passed! Pipeline is working perfectly![/bold green]")
    elif len(successful) > 0:
        console.print("\\n[yellow]⚠️  Some tests failed, but pipeline is partially functional[/yellow]")
    else:
        console.print("\\n[bold red]❌ All tests failed! Please check your configuration[/bold red]")

if __name__ == "__main__":
    app()
