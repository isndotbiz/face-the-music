#!/usr/bin/env python3
"""
Flux Kontext Pro Pipeline with Automatic Error Tracking
Demonstrates how error tracking integrates with your actual pipeline.
"""

import os
import sys
from pathlib import Path
from error_tracker import track_errors, error_tracking_context, _error_tracker
from replicate_generator import ReplicateFluxGenerator
from optimal_pipeline import OptimalPipeline

@track_errors(context="flux_kontext_pipeline", severity="HIGH")
def run_flux_kontext_workflow(input_image: str, face_image: str, prompt: str = "", output_name: str = None):
    """
    Run the complete Flux Kontext Pro workflow with error tracking.
    """
    print("🚀 Starting Flux Kontext Pro Pipeline with Error Tracking")
    print("=" * 60)
    
    # Validate inputs
    if not Path(input_image).exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")
    
    if not Path(face_image).exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")
    
    # Initialize pipeline
    pipeline = OptimalPipeline()
    
    # Run the 4-stage optimal workflow
    with error_tracking_context("optimal_workflow_execution", "HIGH"):
        result = pipeline.process_optimal_workflow(
            input_image_path=input_image,
            face_image_path=face_image,
            prompt=prompt,
            output_name=output_name
        )
    
    if result.success:
        print(f"✅ Pipeline completed successfully!")
        print(f"📁 Final output: {result.final_path}")
        return result.final_path
    else:
        raise Exception(f"Pipeline failed: {result.errors}")

@track_errors(context="flux_generation_only", severity="MEDIUM")
def run_flux_generation_only(prompt: str, face_image: str = None, output_name: str = None):
    """
    Run Flux Kontext Pro generation only (single stage).
    """
    if not os.getenv('REPLICATE_API_TOKEN'):
        raise EnvironmentError("REPLICATE_API_TOKEN not set")
    
    generator = ReplicateFluxGenerator()
    
    # Use the correct model name for Flux Kontext Pro
    image = generator.generate_image(
        prompt=prompt,
        model='flux-kontext-pro',
        reference_image_path=face_image,
        width=1024,
        height=1024,  # Within the 1440 max limit
        steps=28,
        guidance_scale=4.0
    )
    
    if image:
        output_path = f"output/{output_name or 'flux_kontext'}_generated.png"
        Path("output").mkdir(exist_ok=True)
        image.save(output_path)
        print(f"✅ Image generated: {output_path}")
        return output_path
    else:
        raise Exception("Flux generation failed")

def demo_real_pipeline():
    """
    Demo the real pipeline with error tracking.
    """
    print("🎯 Demo: Real Pipeline with Error Tracking")
    print("=" * 50)
    
    # Example 1: Simulate a successful run
    print("\n📝 Example 1: Simulated successful workflow")
    try:
        # This would normally work with real images
        # run_flux_kontext_workflow("input.jpg", "face.jpg", "professional portrait")
        print("Would run: flux_kontext_workflow with real images")
        print("✅ (Simulated success)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Generate error for demo
    print("\n📝 Example 2: File not found error")
    try:
        with error_tracking_context("file_validation", "HIGH"):
            if not os.path.exists("nonexistent_image.jpg"):
                raise FileNotFoundError("Input image file not found: nonexistent_image.jpg")
    except FileNotFoundError as e:
        print(f"⚠️  Caught and tracked: {e}")
    
    # Example 3: API key missing error
    print("\n📝 Example 3: API configuration error")
    try:
        if not os.getenv('REPLICATE_API_TOKEN'):
            with error_tracking_context("api_validation", "HIGH"):
                raise EnvironmentError("REPLICATE_API_TOKEN environment variable not set")
    except EnvironmentError as e:
        print(f"⚠️  Caught and tracked: {e}")
    
    # Show updated error status
    print("\n📊 Error Tracking Status After Demo:")
    summary = _error_tracker.get_error_summary()
    print(f"Active Errors: {summary['active_errors_count']}")
    print(f"Fixed Errors: {summary['fixed_errors_count']}")
    print(f"Total Tracked: {summary['total_errors_tracked']}")

def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Flux Kontext Pro Pipeline with Error Tracking")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--input", help="Input image path")
    parser.add_argument("--face", help="Face reference image path")
    parser.add_argument("--prompt", default="", help="Generation prompt")
    parser.add_argument("--output", help="Output name")
    parser.add_argument("--flux-only", action="store_true", help="Run Flux generation only")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_real_pipeline()
        return
    
    if args.flux_only:
        if not args.prompt:
            print("❌ --prompt required for Flux generation")
            sys.exit(1)
        
        try:
            output_path = run_flux_generation_only(
                prompt=args.prompt,
                face_image=args.face,
                output_name=args.output
            )
            print(f"✅ Generated: {output_path}")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            sys.exit(1)
    
    elif args.input and args.face:
        try:
            output_path = run_flux_kontext_workflow(
                input_image=args.input,
                face_image=args.face,
                prompt=args.prompt,
                output_name=args.output
            )
            print(f"✅ Pipeline completed: {output_path}")
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            sys.exit(1)
    
    else:
        print("Usage examples:")
        print("  # Run demo mode")
        print("  python3 flux_kontext_with_tracking.py --demo")
        print("")
        print("  # Run full pipeline")
        print("  python3 flux_kontext_with_tracking.py --input image.jpg --face face.jpg --prompt 'portrait'")
        print("")
        print("  # Run Flux generation only")
        print("  python3 flux_kontext_with_tracking.py --flux-only --prompt 'portrait' --face face.jpg")
        print("")
        print("  # Check error status")
        print("  python3 error_tracker.py status")

if __name__ == "__main__":
    main()
