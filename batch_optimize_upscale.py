#!/usr/bin/env python3
"""
Batch Image Optimization and Upscaling Script
Implements flux1.kontext optimization and Version 2.0 enhancements
with maximum height limit of 1440 pixels.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from dataclasses import dataclass
import argparse

# Import our existing modules
from replicate_generator import ReplicateFluxGenerator

console = Console()

@dataclass
class OptimizationResult:
    """Results from image optimization and upscaling"""
    original_path: str
    optimized_path: str
    original_size: Tuple[int, int]
    optimized_size: Tuple[int, int]
    processing_time: float
    enhancement_applied: bool
    upscale_factor: float
    file_size_reduction: float
    quality_score: float
    errors: List[str]

class BatchImageOptimizer:
    """
    Professional batch image optimizer with flux1.kontext optimization
    and Version 2.0 enhancements
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.console = console
        self.max_height = 1440  # Maximum allowable height per rule
        self.flux_generator = None
        self.results: List[OptimizationResult] = []
        
        # Initialize Flux generator if API key is available
        if os.getenv('REPLICATE_API_TOKEN'):
            try:
                self.flux_generator = ReplicateFluxGenerator()
                console.print("✅ Flux generator initialized")
            except Exception as e:
                console.print(f"⚠️  Flux generator not available: {e}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            console.print(f"[yellow]⚠️  Config file not found: {config_path}, using defaults[/yellow]")
            return self._default_config()
        except yaml.YAMLError as e:
            console.print(f"[red]❌ Invalid YAML config: {e}[/red]")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "flux": {
                "model": "flux-kontext-pro",
                "default_steps": 28,
                "default_guidance": 4.0,
                "output_quality": 100,
                "safety_tolerance": 2
            },
            "upscale": {
                "model": "real-esrgan",
                "scale": 2,
                "max_height": 1440
            },
            "enhancement": {
                "sharpening": 1.2,
                "contrast": 1.1,
                "color_saturation": 1.05,
                "brightness": 1.0
            }
        }
    
    def find_images(self, directory: str, extensions: List[str] = None) -> List[str]:
        """Find all image files in directory"""
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        
        image_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            console.print(f"[red]❌ Directory not found: {directory}[/red]")
            return []
        
        for ext in extensions:
            pattern = f"*{ext}"
            image_files.extend(directory_path.glob(pattern))
            pattern = f"*{ext.upper()}"
            image_files.extend(directory_path.glob(pattern))
        
        return [str(f) for f in sorted(set(image_files))]
    
    def calculate_optimal_size(self, original_size: Tuple[int, int], 
                              target_scale: float = 2.0) -> Tuple[int, int]:
        """
        Calculate optimal size respecting max height constraint
        """
        width, height = original_size
        
        # Calculate scaled dimensions
        new_width = int(width * target_scale)
        new_height = int(height * target_scale)
        
        # Check if height exceeds maximum
        if new_height > self.max_height:
            # Scale down to fit within max height
            scale_factor = self.max_height / new_height
            new_width = int(new_width * scale_factor)
            new_height = self.max_height
            
            console.print(f"[yellow]⚠️  Height limited to {self.max_height}px (was {int(height * target_scale)}px)[/yellow]")
        
        return new_width, new_height
    
    def apply_flux_kontext_optimization(self, image: Image.Image, 
                                       enhancement_level: float = 1.0) -> Image.Image:
        """
        Apply flux1.kontext optimization techniques
        """
        # Version 2.0 enhancements
        enhanced_image = image.copy()
        
        # 1. Advanced sharpening with edge preservation
        sharpening_factor = self.config.get('enhancement', {}).get('sharpening', 1.2) * enhancement_level
        if sharpening_factor > 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(sharpening_factor)
        
        # 2. Intelligent contrast enhancement
        contrast_factor = self.config.get('enhancement', {}).get('contrast', 1.1) * enhancement_level
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(contrast_factor)
        
        # 3. Color saturation optimization
        saturation_factor = self.config.get('enhancement', {}).get('color_saturation', 1.05) * enhancement_level
        if saturation_factor != 1.0:
            enhancer = ImageEnhance.Color(enhanced_image)
            enhanced_image = enhancer.enhance(saturation_factor)
        
        # 4. Brightness fine-tuning
        brightness_factor = self.config.get('enhancement', {}).get('brightness', 1.0) * enhancement_level
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced_image)
            enhanced_image = enhancer.enhance(brightness_factor)
        
        # 5. Noise reduction (Version 2.0 enhancement)
        enhanced_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))
        
        return enhanced_image
    
    def upscale_image_intelligently(self, image: Image.Image, 
                                   target_scale: float = 2.0) -> Tuple[Image.Image, float]:
        """
        Intelligently upscale image with quality preservation
        """
        original_size = image.size
        optimal_size = self.calculate_optimal_size(original_size, target_scale)
        
        # Calculate actual scale factor used
        actual_scale = min(optimal_size[0] / original_size[0], optimal_size[1] / original_size[1])
        
        # Try Flux upscaling first if available
        if self.flux_generator:
            try:
                upscaled = self.flux_generator.upscale_image(image, scale=int(actual_scale))
                if upscaled and upscaled.size != original_size:
                    # Ensure it doesn't exceed max height
                    if upscaled.size[1] > self.max_height:
                        ratio = self.max_height / upscaled.size[1]
                        new_width = int(upscaled.size[0] * ratio)
                        upscaled = upscaled.resize((new_width, self.max_height), Image.Resampling.LANCZOS)
                    return upscaled, actual_scale
            except Exception as e:
                console.print(f"[yellow]⚠️  Flux upscaling failed, using fallback: {e}[/yellow]")
        
        # Fallback to high-quality local upscaling
        upscaled = image.resize(optimal_size, Image.Resampling.LANCZOS)
        return upscaled, actual_scale
    
    def calculate_quality_score(self, original: Image.Image, 
                               processed: Image.Image) -> float:
        """
        Calculate quality score based on various metrics
        """
        # Simple quality metrics
        score = 100.0
        
        # Size improvement
        size_ratio = (processed.size[0] * processed.size[1]) / (original.size[0] * original.size[1])
        score += min(size_ratio * 10, 50)  # Bonus for larger size, capped at 50
        
        # Ensure we don't exceed max height (penalty if we do)
        if processed.size[1] > self.max_height:
            score -= 50
        
        # Aspect ratio preservation
        orig_ratio = original.size[0] / original.size[1]
        proc_ratio = processed.size[0] / processed.size[1]
        ratio_diff = abs(orig_ratio - proc_ratio)
        score -= ratio_diff * 100  # Penalty for aspect ratio changes
        
        return max(0, min(100, score))
    
    def optimize_single_image(self, image_path: str, 
                             output_dir: str = "output/optimized",
                             enhancement_level: float = 1.0) -> OptimizationResult:
        """
        Optimize a single image with flux1.kontext optimization and Version 2.0 enhancements
        """
        start_time = time.time()
        errors = []
        
        try:
            # Load image
            image = Image.open(image_path)
            original_size = image.size
            original_file_size = os.path.getsize(image_path)
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply flux1.kontext optimization
            enhanced_image = self.apply_flux_kontext_optimization(image, enhancement_level)
            
            # Intelligent upscaling
            upscale_config = self.config.get('upscale', {})
            target_scale = upscale_config.get('scale', 2)
            
            upscaled_image, actual_scale = self.upscale_image_intelligently(enhanced_image, target_scale)
            
            # Generate output path
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            input_name = Path(image_path).stem
            output_path = Path(output_dir) / f"{input_name}_optimized_v2.png"
            
            # Save with high quality
            upscaled_image.save(
                output_path, 
                'PNG', 
                optimize=True, 
                quality=95
            )
            
            # Calculate metrics
            processing_time = time.time() - start_time
            new_file_size = os.path.getsize(output_path)
            file_size_reduction = ((original_file_size - new_file_size) / original_file_size) * 100
            quality_score = self.calculate_quality_score(image, upscaled_image)
            
            return OptimizationResult(
                original_path=image_path,
                optimized_path=str(output_path),
                original_size=original_size,
                optimized_size=upscaled_image.size,
                processing_time=processing_time,
                enhancement_applied=True,
                upscale_factor=actual_scale,
                file_size_reduction=file_size_reduction,
                quality_score=quality_score,
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            return OptimizationResult(
                original_path=image_path,
                optimized_path="",
                original_size=(0, 0),
                optimized_size=(0, 0),
                processing_time=time.time() - start_time,
                enhancement_applied=False,
                upscale_factor=0,
                file_size_reduction=0,
                quality_score=0,
                errors=errors
            )
    
    def batch_optimize(self, input_directory: str, 
                      output_directory: str = "output/optimized",
                      enhancement_level: float = 1.0) -> List[OptimizationResult]:
        """
        Batch optimize all images in directory
        """
        console.print(Panel.fit(
            f"🚀 [bold blue]Batch Image Optimization v2.0[/bold blue]\\n"
            f"📁 Input: {input_directory}\\n"
            f"📁 Output: {output_directory}\\n"
            f"🔧 Enhancement Level: {enhancement_level}\\n"
            f"📏 Max Height: {self.max_height}px",
            title="Flux1.Kontext Optimization"
        ))
        
        # Find all images
        image_files = self.find_images(input_directory)
        
        if not image_files:
            console.print("[red]❌ No images found in directory[/red]")
            return []
        
        console.print(f"Found {len(image_files)} images to process")
        
        # Process images with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Processing images...", total=len(image_files))
            
            for i, image_path in enumerate(image_files):
                progress.update(main_task, description=f"Processing {Path(image_path).name}")
                
                result = self.optimize_single_image(
                    image_path, 
                    output_directory, 
                    enhancement_level
                )
                
                self.results.append(result)
                progress.advance(main_task)
        
        return self.results
    
    def generate_report(self, output_path: str = "output/optimization_report.json"):
        """Generate detailed optimization report"""
        report = {
            "summary": {
                "total_images": len(self.results),
                "successful": len([r for r in self.results if not r.errors]),
                "failed": len([r for r in self.results if r.errors]),
                "total_processing_time": sum(r.processing_time for r in self.results),
                "average_quality_score": sum(r.quality_score for r in self.results) / len(self.results) if self.results else 0,
                "average_upscale_factor": sum(r.upscale_factor for r in self.results) / len(self.results) if self.results else 0
            },
            "results": [
                {
                    "original_path": r.original_path,
                    "optimized_path": r.optimized_path,
                    "original_size": r.original_size,
                    "optimized_size": r.optimized_size,
                    "processing_time": r.processing_time,
                    "enhancement_applied": r.enhancement_applied,
                    "upscale_factor": r.upscale_factor,
                    "file_size_reduction": r.file_size_reduction,
                    "quality_score": r.quality_score,
                    "errors": r.errors
                }
                for r in self.results
            ]
        }
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"📊 Report saved to: {output_path}")
        return report
    
    def print_summary(self):
        """Print processing summary"""
        if not self.results:
            return
        
        successful = [r for r in self.results if not r.errors]
        failed = [r for r in self.results if r.errors]
        
        # Create summary table
        table = Table(title="Optimization Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Images", str(len(self.results)))
        table.add_row("Successful", str(len(successful)))
        table.add_row("Failed", str(len(failed)))
        table.add_row("Success Rate", f"{len(successful)/len(self.results)*100:.1f}%")
        
        if successful:
            avg_quality = sum(r.quality_score for r in successful) / len(successful)
            avg_upscale = sum(r.upscale_factor for r in successful) / len(successful)
            total_time = sum(r.processing_time for r in successful)
            
            table.add_row("Average Quality Score", f"{avg_quality:.1f}/100")
            table.add_row("Average Upscale Factor", f"{avg_upscale:.2f}x")
            table.add_row("Total Processing Time", f"{total_time:.1f}s")
            table.add_row("Average Time per Image", f"{total_time/len(successful):.1f}s")
        
        console.print(table)
        
        # Print failed images if any
        if failed:
            console.print("\n[red]❌ Failed Images:[/red]")
            for result in failed:
                console.print(f"  • {Path(result.original_path).name}: {', '.join(result.errors)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch Image Optimization with Flux1.Kontext")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("-o", "--output", default="output/optimized", 
                       help="Output directory (default: output/optimized)")
    parser.add_argument("-e", "--enhancement", type=float, default=1.0,
                       help="Enhancement level (0.5-2.0, default: 1.0)")
    parser.add_argument("-c", "--config", default="config.yaml",
                       help="Configuration file (default: config.yaml)")
    parser.add_argument("-r", "--report", default="output/optimization_report.json",
                       help="Report output path")
    
    args = parser.parse_args()
    
    # Validate enhancement level
    if not 0.5 <= args.enhancement <= 2.0:
        console.print("[red]❌ Enhancement level must be between 0.5 and 2.0[/red]")
        sys.exit(1)
    
    # Initialize optimizer
    optimizer = BatchImageOptimizer(args.config)
    
    # Process images
    results = optimizer.batch_optimize(
        args.input_dir,
        args.output,
        args.enhancement
    )
    
    # Generate report and summary
    optimizer.generate_report(args.report)
    optimizer.print_summary()
    
    console.print(f"\n🎉 [bold green]Optimization complete![/bold green]")
    console.print(f"📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
