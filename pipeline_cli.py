#!/usr/bin/env python3
"""
Enhanced CLI Interface for Ultra-Realistic Face Swap Pipeline
Professional command-line interface with preset management

Version: 2.1-PROFESSIONAL
Author: Face The Music Team
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import print as rprint

# Import our professional pipeline
from professional_pipeline import ProfessionalPipeline

console = Console()

class PresetManager:
    """Manage processing presets for different use cases"""
    
    def __init__(self):
        self.presets_file = "pipeline_presets.yaml"
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict[str, Any]:
        """Load presets from file or create defaults"""
        if Path(self.presets_file).exists():
            with open(self.presets_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Create default presets
            defaults = {
                "ultra_quality": {
                    "name": "Ultra Quality",
                    "description": "Maximum quality, longer processing time",
                    "stages": {
                        "stage1": {"quality": "maximum", "steps": 75},
                        "stage2": {"scale": 4, "enhance": True},
                        "stage3": {"strength": 0.25, "steps": 8},
                        "stage4": {"color_grading": True, "film_grain": True}
                    }
                },
                "balanced": {
                    "name": "Balanced",
                    "description": "Good quality with reasonable processing time",
                    "stages": {
                        "stage1": {"quality": "high", "steps": 50},
                        "stage2": {"scale": 2, "enhance": True},
                        "stage3": {"strength": 0.35, "steps": 4},
                        "stage4": {"color_grading": True, "film_grain": False}
                    }
                },
                "fast": {
                    "name": "Fast Processing", 
                    "description": "Quick results for testing and previews",
                    "stages": {
                        "stage1": {"quality": "standard", "steps": 25},
                        "stage2": {"scale": 2, "enhance": False},
                        "stage3": {"strength": 0.45, "steps": 2},
                        "stage4": {"color_grading": False, "film_grain": False}
                    }
                },
                "cinema_grade": {
                    "name": "Cinema Grade",
                    "description": "Professional cinema quality with all enhancements",
                    "stages": {
                        "stage1": {"quality": "cinema", "steps": 100},
                        "stage2": {"scale": 4, "enhance": True},
                        "stage3": {"strength": 0.20, "steps": 12},
                        "stage4": {"color_grading": True, "film_grain": True, "hdr": True}
                    }
                }
            }
            self._save_presets(defaults)
            return defaults
    
    def _save_presets(self, presets: Dict[str, Any]):
        """Save presets to file"""
        with open(self.presets_file, 'w') as f:
            yaml.dump(presets, f, default_flow_style=False)
    
    def list_presets(self) -> List[str]:
        """List available preset names"""
        return list(self.presets.keys())
    
    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get preset by name"""
        return self.presets.get(name)
    
    def add_preset(self, name: str, preset: Dict[str, Any]):
        """Add new preset"""
        self.presets[name] = preset
        self._save_presets(self.presets)

@click.group()
@click.version_option(version="2.1-PROFESSIONAL", prog_name="Face The Music Pipeline")
def cli():
    """
    🎬 Ultra-Realistic Face Swap & Enhancement Pipeline
    
    Professional-grade multi-stage image processing for cinema-quality results.
    """
    pass

@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), 
              help='Input image path')
@click.option('--face', '-f', required=True, type=click.Path(exists=True),
              help='Face reference image path')
@click.option('--output', '-o', help='Custom output name (auto-generated if not provided)')
@click.option('--prompt', '-p', default='', help='Enhancement prompt for generation')
@click.option('--preset', '-s', help='Processing preset (ultra_quality, balanced, fast, cinema_grade)')
@click.option('--config', '-c', default='workflow_config.yaml', 
              type=click.Path(exists=True), help='Configuration file path')
@click.option('--preview', is_flag=True, help='Show processing preview before starting')
@click.option('--batch', type=int, help='Process multiple variations (1-10)')
def process(input, face, output, prompt, preset, config, preview, batch):
    """
    🚀 Process image through the professional pipeline
    
    This command runs the complete 4-stage ultra-realistic face swap pipeline
    with optional preset configurations for different quality/speed trade-offs.
    """
    
    # Display header
    console.print(Panel.fit(
        "🎬 [bold blue]Ultra-Realistic Face Swap Pipeline v2.1[/bold blue]\n"
        "Professional-grade multi-stage processing",
        title="Face The Music"
    ))
    
    # Validate and display inputs
    input_path = Path(input)
    face_path = Path(face)
    
    info_table = Table(title="📋 Processing Configuration")
    info_table.add_column("Parameter", style="cyan")
    info_table.add_column("Value", style="green")
    
    info_table.add_row("Input Image", str(input_path.name))
    info_table.add_row("Face Reference", str(face_path.name))
    info_table.add_row("Output Name", output or "auto-generated")
    info_table.add_row("Prompt", prompt or "default professional")
    info_table.add_row("Preset", preset or "default")
    info_table.add_row("Config File", config)
    if batch:
        info_table.add_row("Batch Size", str(batch))
    
    console.print(info_table)
    
    # Handle preset selection
    preset_manager = PresetManager()
    if preset:
        if preset not in preset_manager.list_presets():
            console.print(f"[red]❌ Unknown preset: {preset}[/red]")
            console.print("Available presets:", ", ".join(preset_manager.list_presets()))
            sys.exit(1)
        
        preset_config = preset_manager.get_preset(preset)
        console.print(f"[green]✅ Using preset: {preset_config['name']}[/green]")
        console.print(f"   Description: {preset_config['description']}")
    
    # Preview mode
    if preview:
        console.print("\n[yellow]🔍 Preview Mode - Processing Details:[/yellow]")
        
        preview_table = Table(title="🎭 Processing Stages")
        preview_table.add_column("Stage", style="cyan")
        preview_table.add_column("Process", style="yellow")
        preview_table.add_column("Expected Time", style="magenta")
        
        preview_table.add_row("1", "Advanced Face Swap (Flux Kontext Pro)", "30-60s")
        preview_table.add_row("2", "Neural Upscaling (Real-ESRGAN)", "20-45s")
        preview_table.add_row("3", "AI Refinement (SDXL Turbo)", "15-30s")
        preview_table.add_row("4", "Cinema Post-Processing", "5-10s")
        
        console.print(preview_table)
        
        if not Confirm.ask("\n🚀 Start processing?"):
            console.print("[yellow]⏹️  Processing cancelled[/yellow]")
            return
    
    # Process single image or batch
    try:
        pipeline = ProfessionalPipeline(config)
        
        if batch and batch > 1:
            console.print(f"\n[blue]🔄 Processing {batch} variations...[/blue]")
            results = []
            
            for i in range(batch):
                console.print(f"\n[cyan]📸 Processing variation {i+1}/{batch}[/cyan]")
                
                variation_output = f"{output}_var{i+1}" if output else None
                variation_prompt = f"{prompt}, variation {i+1}" if prompt else f"professional portrait variation {i+1}"
                
                result = pipeline.process_image(
                    input_image_path=str(input_path),
                    face_image_path=str(face_path),
                    prompt=variation_prompt,
                    output_name=variation_output
                )
                results.append(result)
            
            # Display batch summary
            _display_batch_summary(results)
            
        else:
            # Single image processing
            result = pipeline.process_image(
                input_image_path=str(input_path),
                face_image_path=str(face_path),
                prompt=prompt,
                output_name=output
            )
            
            if result["status"] == "success":
                console.print("\n[bold green]🎉 Processing completed successfully![/bold green]")
            else:
                console.print(f"\n[bold red]❌ Processing failed: {result.get('error')}[/bold red]")
                sys.exit(1)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Unexpected error: {e}[/bold red]")
        sys.exit(1)

@cli.command()
def presets():
    """📋 List available processing presets"""
    
    preset_manager = PresetManager()
    presets = preset_manager.presets
    
    if not presets:
        console.print("[yellow]No presets available[/yellow]")
        return
    
    table = Table(title="🎨 Available Processing Presets")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Quality", style="magenta")
    table.add_column("Speed", style="yellow")
    
    quality_map = {
        "cinema": "🏆 Cinema",
        "maximum": "⭐ Ultra",
        "high": "✨ High", 
        "standard": "📋 Standard"
    }
    
    speed_map = {
        "cinema": "🐌 Slow",
        "maximum": "🐌 Slow",
        "high": "⚡ Medium",
        "standard": "🚀 Fast"
    }
    
    for name, preset in presets.items():
        quality_level = preset['stages']['stage1'].get('quality', 'standard')
        quality_display = quality_map.get(quality_level, quality_level)
        speed_display = speed_map.get(quality_level, "⚡ Medium")
        
        table.add_row(
            name,
            preset['description'],
            quality_display,
            speed_display
        )
    
    console.print(table)
    
    console.print("\n[dim]💡 Use with: --preset <name>[/dim]")

@cli.command()
def status():
    """📊 Check system status and requirements"""
    
    console.print(Panel.fit(
        "🔍 [bold blue]System Status Check[/bold blue]",
        title="Face The Music Pipeline"
    ))
    
    status_table = Table(title="🖥️  System Information")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="yellow")
    
    # Check Python version
    python_version = sys.version.split()[0]
    python_ok = tuple(map(int, python_version.split('.'))) >= (3, 8)
    status_table.add_row(
        "Python Version",
        "✅ OK" if python_ok else "❌ Outdated",
        f"{python_version} (minimum: 3.8)"
    )
    
    # Check API token
    api_token = os.getenv('REPLICATE_API_TOKEN')
    status_table.add_row(
        "Replicate API",
        "✅ Configured" if api_token else "❌ Missing",
        "Token set" if api_token else "Set REPLICATE_API_TOKEN"
    )
    
    # Check dependencies
    required_deps = ['replicate', 'PIL', 'rich', 'click', 'yaml']
    missing_deps = []
    
    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    status_table.add_row(
        "Dependencies",
        "✅ Complete" if not missing_deps else "❌ Missing",
        "All installed" if not missing_deps else f"Missing: {', '.join(missing_deps)}"
    )
    
    # Check GPU availability
    gpu_available = False
    gpu_info = "Not available"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_count = torch.cuda.device_count()
            gpu_info = f"{gpu_count} GPU(s) available"
    except ImportError:
        gpu_info = "PyTorch not installed"
    
    status_table.add_row(
        "GPU Acceleration",
        "✅ Available" if gpu_available else "⚠️  CPU Only",
        gpu_info
    )
    
    # Check disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        space_ok = free_space > 5  # Need at least 5GB
        status_table.add_row(
            "Disk Space",
            "✅ Sufficient" if space_ok else "⚠️  Low",
            f"{free_space:.1f} GB free"
        )
    except Exception:
        status_table.add_row(
            "Disk Space",
            "❓ Unknown",
            "Unable to check"
        )
    
    console.print(status_table)
    
    # Overall status
    overall_ok = python_ok and api_token and not missing_deps
    if overall_ok:
        console.print("\n[bold green]🎉 System ready for professional processing![/bold green]")
    else:
        console.print("\n[bold yellow]⚠️  Some issues detected. Please resolve them before processing.[/bold yellow]")

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--face', '-f', required=True, type=click.Path(exists=True),
              help='Face reference image path')
@click.option('--output-dir', '-o', default='output/batch',
              help='Output directory for batch results')
@click.option('--preset', '-s', default='balanced', help='Processing preset to use')
@click.option('--prompt', '-p', default='', help='Base prompt for all images')
@click.option('--max-count', '-n', type=int, help='Maximum number of images to process')
def batch(directory, face, output_dir, preset, prompt, max_count):
    """
    🔄 Batch process all images in a directory
    
    Processes multiple images through the pipeline using the same face reference.
    Useful for processing entire photo collections or datasets.
    """
    
    directory = Path(directory)
    face_path = Path(face)
    output_path = Path(output_dir)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        console.print(f"[red]❌ No image files found in {directory}[/red]")
        return
    
    if max_count:
        image_files = image_files[:max_count]
    
    console.print(f"[blue]🔍 Found {len(image_files)} images to process[/blue]")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Confirm batch processing
    if not Confirm.ask(f"\n🚀 Process {len(image_files)} images with preset '{preset}'?"):
        console.print("[yellow]⏹️  Batch processing cancelled[/yellow]")
        return
    
    # Initialize pipeline
    try:
        pipeline = ProfessionalPipeline()
        results = []
        
        console.print(f"\n[blue]🔄 Starting batch processing...[/blue]")
        
        for i, image_file in enumerate(image_files, 1):
            console.print(f"\n[cyan]📸 Processing {i}/{len(image_files)}: {image_file.name}[/cyan]")
            
            # Generate output name
            output_name = f"batch_{image_file.stem}_{int(time.time())}"
            
            try:
                result = pipeline.process_image(
                    input_image_path=str(image_file),
                    face_image_path=str(face_path),
                    prompt=f"{prompt}, professional photo {i}" if prompt else f"professional photo {i}",
                    output_name=output_name
                )
                results.append((image_file.name, result))
                
                if result["status"] == "success":
                    console.print(f"[green]✅ Completed: {image_file.name}[/green]")
                else:
                    console.print(f"[red]❌ Failed: {image_file.name} - {result.get('error')}[/red]")
                    
            except Exception as e:
                console.print(f"[red]❌ Error processing {image_file.name}: {e}[/red]")
                results.append((image_file.name, {"status": "error", "error": str(e)}))
        
        # Display batch summary
        _display_batch_summary(results)
        
    except Exception as e:
        console.print(f"[bold red]❌ Batch processing failed: {e}[/bold red]")
        sys.exit(1)

def _display_batch_summary(results):
    """Display summary of batch processing results"""
    
    successful = sum(1 for _, result in results if isinstance(result, dict) and result.get("status") == "success")
    failed = len(results) - successful
    
    summary_table = Table(title="📊 Batch Processing Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Images", str(len(results)))
    summary_table.add_row("Successful", str(successful))
    summary_table.add_row("Failed", str(failed))
    summary_table.add_row("Success Rate", f"{(successful/len(results)*100):.1f}%")
    
    console.print(summary_table)
    
    # Show detailed results
    if failed > 0:
        console.print("\n[yellow]⚠️  Failed Images:[/yellow]")
        for name, result in results:
            if isinstance(result, dict) and result.get("status") == "error":
                console.print(f"   [red]❌ {name}: {result.get('error', 'Unknown error')}[/red]")

if __name__ == "__main__":
    cli()
