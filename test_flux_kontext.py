#!/usr/bin/env python3
"""
Simple test script to verify Flux Kontext Pro face swapping works
"""

import os
import sys
from pathlib import Path
from replicate_generator import ReplicateFluxGenerator
from rich.console import Console
from rich.panel import Panel
import time

console = Console()

def main():
    console.print(Panel.fit(
        "🎭 [bold blue]Testing Flux Kontext Pro Face Swapping[/bold blue]\n"
        "Simple test to verify the system works",
        title="Face The Music Test"
    ))
    
    # Check if API token is set
    if not os.getenv('REPLICATE_API_TOKEN'):
        console.print("[red]❌ REPLICATE_API_TOKEN not set![/red]")
        console.print("Set it with: export REPLICATE_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Check if face image exists
    face_path = Path("faces/your_face.png")
    if not face_path.exists():
        console.print(f"[red]❌ Face image not found: {face_path}[/red]")
        sys.exit(1)
    
    try:
        # Initialize generator
        console.print("🔧 Initializing Flux generator...")
        generator = ReplicateFluxGenerator()
        
        # Generate a test image
        console.print("🎨 Generating test image with Flux Kontext Pro...")
        
        prompt = (
            "professional portrait on luxury yacht deck, Mediterranean sunset, "
            "sophisticated lighting, high quality, photorealistic, 8K"
        )
        
        console.print(f"📝 Prompt: {prompt}")
        console.print(f"👤 Face reference: {face_path}")
        
        start_time = time.time()
        
        result_image = generator.generate_image(
            prompt=prompt,
            width=1024,
            height=1024,
            model='flux-kontext-pro',
            reference_image_path=str(face_path),
            steps=25,
            guidance_scale=3.5
        )
        
        elapsed = time.time() - start_time
        
        if result_image:
            # Save the result
            output_path = f"output/test_flux_kontext_{int(time.time())}.png"
            result_image.save(output_path)
            
            console.print(f"[green]✅ Success! Generated in {elapsed:.1f}s[/green]")
            console.print(f"📁 Saved to: {output_path}")
            console.print(f"📏 Size: {result_image.size}")
            
            # Try upscaling
            console.print("\n🔍 Testing upscaling...")
            upscaled = generator.upscale_image(result_image, scale=2)
            
            if upscaled:
                upscaled_path = f"output/test_flux_kontext_upscaled_{int(time.time())}.png"
                upscaled.save(upscaled_path)
                console.print(f"[green]✅ Upscaled successfully![/green]")
                console.print(f"📁 Upscaled saved to: {upscaled_path}")
                console.print(f"📏 Upscaled size: {upscaled.size}")
            else:
                console.print("[yellow]⚠️  Upscaling failed[/yellow]")
            
        else:
            console.print("[red]❌ Generation failed![/red]")
            sys.exit(1)
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)
    
    console.print("\n[bold green]🎉 Test completed successfully![/bold green]")

if __name__ == "__main__":
    main()
