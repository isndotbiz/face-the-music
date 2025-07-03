#!/usr/bin/env python3
"""
Quick Image Optimization Tool
Optimizes and upscales images with flux1.kontext optimization and Version 2.0 enhancements
while ensuring images do not exceed 1440px height limit.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List
from PIL import Image, ImageEnhance, ImageFilter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import argparse

console = Console()

class ImageOptimizer:
    """Quick image optimizer with height constraint enforcement"""
    
    def __init__(self, max_height: int = 1440):
        self.max_height = max_height
        
    def check_image_dimensions(self, image_path: str) -> Tuple[bool, Tuple[int, int]]:
        """Check if image dimensions comply with height constraint"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                return height <= self.max_height, (width, height)
        except Exception as e:
            console.print(f"[red]❌ Error reading {image_path}: {e}[/red]")
            return False, (0, 0)
    
    def resize_to_constraint(self, image: Image.Image, target_scale: float = 2.0) -> Image.Image:
        """Resize image respecting height constraint"""
        width, height = image.size
        
        # Calculate new dimensions
        new_width = int(width * target_scale)
        new_height = int(height * target_scale)
        
        # Check height constraint
        if new_height > self.max_height:
            # Calculate scale factor to fit within max height
            scale_factor = self.max_height / new_height
            new_width = int(new_width * scale_factor)
            new_height = self.max_height
            console.print(f"[yellow]📏 Resized to fit height constraint: {new_width}x{new_height}[/yellow]")
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def apply_flux_kontext_optimizations(self, image: Image.Image) -> Image.Image:
        """Apply flux1.kontext optimization and Version 2.0 enhancements"""
        enhanced = image.copy()
        
        # 1. Sharpening enhancement
        sharpness = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpness.enhance(1.2)
        
        # 2. Contrast optimization
        contrast = ImageEnhance.Contrast(enhanced)
        enhanced = contrast.enhance(1.1)
        
        # 3. Color saturation boost
        color = ImageEnhance.Color(enhanced)
        enhanced = color.enhance(1.05)
        
        # 4. Noise reduction (Version 2.0)
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
        
        # 5. Additional sharpening pass
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        return enhanced
    
    def optimize_image(self, input_path: str, output_path: str = None, 
                      upscale_factor: float = 2.0) -> bool:
        """Optimize a single image"""
        try:
            # Load image
            with Image.open(input_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                console.print(f"📸 Processing: {Path(input_path).name}")
                console.print(f"📏 Original size: {img.size}")
                
                # Apply flux1.kontext optimizations
                optimized = self.apply_flux_kontext_optimizations(img)
                
                # Upscale with height constraint
                upscaled = self.resize_to_constraint(optimized, upscale_factor)
                
                console.print(f"📏 Final size: {upscaled.size}")
                
                # Validate height constraint
                if upscaled.size[1] > self.max_height:
                    console.print(f"[red]❌ Height constraint violated: {upscaled.size[1]} > {self.max_height}[/red]")
                    return False
                
                # Generate output path if not provided
                if not output_path:
                    input_path_obj = Path(input_path)
                    output_path = input_path_obj.parent / f"{input_path_obj.stem}_optimized_v2.png"
                
                # Save optimized image
                upscaled.save(output_path, 'PNG', optimize=True, quality=95)
                
                console.print(f"[green]✅ Saved to: {output_path}[/green]")
                return True
                
        except Exception as e:
            console.print(f"[red]❌ Error processing {input_path}: {e}[/red]")
            return False
    
    def batch_optimize(self, input_dir: str, output_dir: str = None, 
                      upscale_factor: float = 2.0) -> List[str]:
        """Batch optimize images in directory"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            console.print(f"[red]❌ Input directory not found: {input_dir}[/red]")
            return []
        
        # Find image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        if not image_files:
            console.print("[red]❌ No image files found[/red]")
            return []
        
        # Set up output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = input_path / "optimized"
        
        output_path.mkdir(exist_ok=True)
        
        console.print(Panel.fit(
            f"🚀 [bold blue]Batch Image Optimization[/bold blue]\\n"
            f"📁 Input: {input_dir}\\n"
            f"📁 Output: {output_path}\\n"
            f"🔍 Upscale Factor: {upscale_factor}x\\n"
            f"📏 Max Height: {self.max_height}px\\n"
            f"📸 Found {len(image_files)} images",
            title="Flux1.Kontext Optimization v2.0"
        ))
        
        successful = []
        failed = []
        
        for image_file in image_files:
            output_file = output_path / f"{image_file.stem}_optimized_v2.png"
            
            if self.optimize_image(str(image_file), str(output_file), upscale_factor):
                successful.append(str(output_file))
            else:
                failed.append(str(image_file))
        
        # Print summary
        table = Table(title="Optimization Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Total Images", str(len(image_files)))
        table.add_row("Successful", str(len(successful)))
        table.add_row("Failed", str(len(failed)))
        table.add_row("Success Rate", f"{len(successful)/len(image_files)*100:.1f}%")
        
        console.print(table)
        
        if failed:
            console.print("\\n[red]❌ Failed files:[/red]")
            for fail in failed:
                console.print(f"  • {Path(fail).name}")
        
        return successful
    
    def validate_directory(self, directory: str) -> Tuple[List[str], List[str]]:
        """Validate all images in directory against height constraint"""
        compliant = []
        non_compliant = []
        
        input_path = Path(directory)
        if not input_path.exists():
            console.print(f"[red]❌ Directory not found: {directory}[/red]")
            return [], []
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                is_compliant, dimensions = self.check_image_dimensions(str(file_path))
                
                if is_compliant:
                    compliant.append(str(file_path))
                else:
                    non_compliant.append(str(file_path))
        
        return compliant, non_compliant


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Image Optimization with Height Constraint")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-s", "--scale", type=float, default=2.0,
                       help="Upscale factor (default: 2.0)")
    parser.add_argument("--max-height", type=int, default=1440,
                       help="Maximum height in pixels (default: 1440)")
    parser.add_argument("--validate", action="store_true",
                       help="Only validate images against height constraint")
    parser.add_argument("--batch", action="store_true",
                       help="Process directory in batch mode")
    
    args = parser.parse_args()
    
    # Validate scale factor
    if args.scale < 0.5 or args.scale > 4.0:
        console.print("[red]❌ Scale factor must be between 0.5 and 4.0[/red]")
        sys.exit(1)
    
    optimizer = ImageOptimizer(max_height=args.max_height)
    
    if args.validate:
        # Validation mode
        if Path(args.input).is_dir():
            compliant, non_compliant = optimizer.validate_directory(args.input)
            
            console.print(Panel.fit(
                f"📏 [bold blue]Height Constraint Validation[/bold blue]\\n"
                f"📁 Directory: {args.input}\\n"
                f"📏 Max Height: {args.max_height}px\\n"
                f"✅ Compliant: {len(compliant)}\\n"
                f"❌ Non-compliant: {len(non_compliant)}",
                title="Image Validation Results"
            ))
            
            if non_compliant:
                console.print("\\n[red]❌ Non-compliant images:[/red]")
                for img in non_compliant:
                    console.print(f"  • {Path(img).name}")
        else:
            is_compliant, dimensions = optimizer.check_image_dimensions(args.input)
            if is_compliant:
                console.print(f"[green]✅ {args.input} is compliant ({dimensions[0]}x{dimensions[1]})[/green]")
            else:
                console.print(f"[red]❌ {args.input} exceeds height limit ({dimensions[0]}x{dimensions[1]})[/red]")
    
    elif args.batch or Path(args.input).is_dir():
        # Batch mode
        optimizer.batch_optimize(args.input, args.output, args.scale)
    
    else:
        # Single file mode
        if optimizer.optimize_image(args.input, args.output, args.scale):
            console.print(f"[green]🎉 Optimization complete![/green]")
        else:
            console.print(f"[red]❌ Optimization failed![/red]")
            sys.exit(1)


if __name__ == "__main__":
    main()
