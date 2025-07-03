#!/usr/bin/env python3
"""
Batch Optimal Pipeline Processor
Processes multiple images through the optimal flux1.kontext workflow
with intelligent height management and LoRA integration.

Author: Face The Music Team
Version: 2.0-BATCH-OPTIMAL
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from optimal_pipeline import OptimalPipeline, WorkflowResult

console = Console()

class BatchOptimalProcessor:
    """Batch processor for optimal pipeline"""
    
    def __init__(self, config_path: str = "workflow_config.yaml"):
        self.pipeline = OptimalPipeline(config_path)
        self.console = console
        self.results: List[WorkflowResult] = []
    
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
    
    def validate_face_reference(self, face_path: str) -> bool:
        """Validate face reference image exists and is valid"""
        face_file = Path(face_path)
        if not face_file.exists():
            console.print(f"[red]❌ Face reference not found: {face_path}[/red]")
            return False
        
        try:
            from PIL import Image
            with Image.open(face_path) as img:
                if img.size[0] < 256 or img.size[1] < 256:
                    console.print(f"[yellow]⚠️  Face reference is quite small: {img.size}[/yellow]")
                return True
        except Exception as e:
            console.print(f"[red]❌ Invalid face reference: {e}[/red]")
            return False
    
    def process_batch(self, 
                     input_directory: str,
                     face_reference: str,
                     prompt: str = "",
                     output_prefix: str = "batch") -> List[WorkflowResult]:
        """Process all images in directory through optimal pipeline"""
        
        # Validate face reference
        if not self.validate_face_reference(face_reference):
            return []
        
        # Find images
        image_files = self.find_images(input_directory)
        
        if not image_files:
            console.print("[red]❌ No images found in directory[/red]")
            return []
        
        console.print(Panel.fit(
            f"🚀 [bold blue]Batch Optimal Pipeline v2.0[/bold blue]\\n"
            f"📁 Input Directory: {input_directory}\\n"
            f"👤 Face Reference: {Path(face_reference).name}\\n"
            f"📸 Found Images: {len(image_files)}\\n"
            f"💬 Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}\\n"
            f"📏 Max Height: 1440px (Flux Constraint)",
            title="Batch Processing with Flux1.Kontext Optimization"
        ))
        
        # Process each image
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Processing batch...", total=len(image_files))
            
            for i, image_path in enumerate(image_files):
                image_name = Path(image_path).stem
                output_name = f"{output_prefix}_{i+1:03d}_{image_name}"
                
                progress.update(main_task, description=f"Processing {Path(image_path).name}")
                
                # Process single image
                result = self.pipeline.process_optimal_workflow(
                    input_image_path=image_path,
                    face_image_path=face_reference,
                    prompt=prompt,
                    output_name=output_name
                )
                
                self.results.append(result)
                progress.advance(main_task)
        
        return self.results
    
    def generate_batch_report(self, output_path: str = "output/workflow_reports/batch_report.json"):
        """Generate comprehensive batch processing report"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        # Calculate statistics
        total_time = sum(r.total_time for r in self.results)
        avg_time = total_time / len(self.results) if self.results else 0
        
        # Quality metrics
        quality_scores = []
        for result in successful:
            if result.quality_metrics and "structural_similarity" in result.quality_metrics:
                quality_scores.append(result.quality_metrics["structural_similarity"])
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Size analysis
        size_improvements = []
        for result in successful:
            if result.original_size and result.final_size:
                orig_pixels = result.original_size[0] * result.original_size[1]
                final_pixels = result.final_size[0] * result.final_size[1]
                improvement = (final_pixels / orig_pixels) if orig_pixels > 0 else 1
                size_improvements.append(improvement)
        
        avg_size_improvement = sum(size_improvements) / len(size_improvements) if size_improvements else 1
        
        report = {
            "batch_summary": {
                "total_images": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": (len(successful) / len(self.results)) * 100 if self.results else 0,
                "total_processing_time": total_time,
                "average_time_per_image": avg_time,
                "average_quality_score": avg_quality,
                "average_size_improvement": avg_size_improvement
            },
            "successful_results": [
                {
                    "input_file": Path(r.original_path).name,
                    "output_file": Path(r.final_path).name,
                    "original_size": r.original_size,
                    "final_size": r.final_size,
                    "processing_time": r.total_time,
                    "quality_score": r.quality_metrics.get("structural_similarity", 0) if r.quality_metrics else 0
                }
                for r in successful
            ],
            "failed_results": [
                {
                    "input_file": Path(r.original_path).name,
                    "errors": r.errors,
                    "processing_time": r.total_time
                }
                for r in failed
            ],
            "workflow_configuration": {
                "flux_height_limit": 1440,
                "stages": [
                    "Flux Kontext Pro Face Swap",
                    "SDXL + Photorealism LoRAs", 
                    "Intelligent Enhancement & Upscaling",
                    "Professional Post-Processing"
                ]
            }
        }
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"📊 Batch report saved to: {output_path}")
        return report
    
    def print_batch_summary(self):
        """Print batch processing summary"""
        if not self.results:
            console.print("[yellow]No results to display[/yellow]")
            return
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        # Main summary table
        table = Table(title="🎯 Batch Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Images", str(len(self.results)))
        table.add_row("Successful", f"{len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        table.add_row("Failed", f"{len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")
        
        if successful:
            total_time = sum(r.total_time for r in successful)
            avg_time = total_time / len(successful)
            
            # Calculate quality metrics
            quality_scores = []
            for result in successful:
                if result.quality_metrics and "structural_similarity" in result.quality_metrics:
                    quality_scores.append(result.quality_metrics["structural_similarity"])
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                table.add_row("Average Quality", f"{avg_quality:.3f}")
            
            table.add_row("Total Processing Time", f"{total_time:.1f}s")
            table.add_row("Average Time per Image", f"{avg_time:.1f}s")
            
            # Height compliance check
            height_compliant = 0
            for result in successful:
                if result.final_size and result.final_size[1] <= 1440:
                    height_compliant += 1
            
            table.add_row("Height Compliant", f"{height_compliant}/{len(successful)} (≤1440px)")
        
        console.print(table)
        
        # Failed images details
        if failed:
            console.print("\n[red]❌ Failed Images:[/red]")
            for result in failed:
                console.print(f"  • {Path(result.original_path).name}: {', '.join(result.errors)}")
        
        # Success details
        if successful:
            console.print("\n[green]✅ Successful Images:[/green]")
            for result in successful[:5]:  # Show first 5
                size_info = f"{result.original_size} → {result.final_size}" if result.final_size else "N/A"
                console.print(f"  • {Path(result.original_path).name}: {size_info} ({result.total_time:.1f}s)")
            
            if len(successful) > 5:
                console.print(f"  ... and {len(successful) - 5} more")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch Optimal Pipeline Processor v2.0")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("face", help="Face reference image path")
    parser.add_argument("-p", "--prompt", default="", help="Enhancement prompt for all images")
    parser.add_argument("-o", "--output-prefix", default="batch", help="Output filename prefix")
    parser.add_argument("-c", "--config", default="workflow_config.yaml", help="Config file")
    parser.add_argument("-r", "--report", default="output/workflow_reports/batch_report.json",
                       help="Batch report output path")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input_dir).exists():
        console.print(f"[red]❌ Input directory not found: {args.input_dir}[/red]")
        sys.exit(1)
    
    if not Path(args.face).exists():
        console.print(f"[red]❌ Face reference not found: {args.face}[/red]")
        sys.exit(1)
    
    # Initialize batch processor
    processor = BatchOptimalProcessor(args.config)
    
    # Process batch
    results = processor.process_batch(
        args.input_dir,
        args.face,
        args.prompt,
        args.output_prefix
    )
    
    if results:
        # Generate report and summary
        processor.generate_batch_report(args.report)
        processor.print_batch_summary()
        
        successful = [r for r in results if r.success]
        console.print(f"\n🎉 [bold green]Batch processing complete![/bold green]")
        console.print(f"✅ {len(successful)}/{len(results)} images processed successfully")
        
        if successful:
            console.print(f"📁 Outputs saved in: output/stage4_final_optimal/")
            console.print(f"📊 Report saved to: {args.report}")
    else:
        console.print("[red]❌ Batch processing failed to start![/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
