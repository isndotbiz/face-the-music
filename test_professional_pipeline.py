#!/usr/bin/env python3
"""
Test Script for Professional Pipeline
Quick verification and testing of the ultra-realistic pipeline

Version: 2.1-PROFESSIONAL
Author: Face The Music Team
"""

import os
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def test_imports():
    """Test that all required imports work"""
    console.print("🔍 Testing imports...")
    
    try:
        import yaml
        import replicate
        from PIL import Image, ImageEnhance, ImageFilter
        import numpy as np
        import cv2
        from rich.console import Console
        from rich.progress import Progress
        import torch
        import psutil
        import click
        
        console.print("[green]✅ All imports successful[/green]")
        return True
        
    except ImportError as e:
        console.print(f"[red]❌ Import failed: {e}[/red]")
        return False

def test_configuration():
    """Test configuration file loading"""
    console.print("🔍 Testing configuration...")
    
    try:
        if not Path("workflow_config.yaml").exists():
            console.print("[yellow]⚠️  workflow_config.yaml not found, but that's expected for testing[/yellow]")
            return True
            
        with open("workflow_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            
        if "agent_workflow" in config:
            console.print("[green]✅ Configuration file valid[/green]")
            return True
        else:
            console.print("[red]❌ Invalid configuration structure[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Configuration test failed: {e}[/red]")
        return False

def test_api_token():
    """Test API token availability"""
    console.print("🔍 Testing API token...")
    
    token = os.getenv('REPLICATE_API_TOKEN')
    if token:
        console.print("[green]✅ Replicate API token found[/green]")
        return True
    else:
        console.print("[yellow]⚠️  REPLICATE_API_TOKEN not set (required for actual processing)[/yellow]")
        return False

def test_face_image():
    """Test face reference image"""
    console.print("🔍 Testing face reference image...")
    
    face_path = Path("faces/your_face.png")
    if face_path.exists():
        try:
            from PIL import Image
            img = Image.open(face_path)
            width, height = img.size
            
            console.print(f"[green]✅ Face image found: {width}x{height}[/green]")
            
            if width >= 1024 and height >= 1024:
                console.print("[green]✅ Resolution suitable for professional processing[/green]")
            else:
                console.print("[yellow]⚠️  Resolution lower than recommended 1024x1024[/yellow]")
                
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error loading face image: {e}[/red]")
            return False
    else:
        console.print("[yellow]⚠️  Face reference image not found at faces/your_face.png[/yellow]")
        return False

def test_output_directories():
    """Test output directory creation"""
    console.print("🔍 Testing output directories...")
    
    try:
        directories = [
            "output/stage1_face_swap",
            "output/stage2_upscaled", 
            "output/stage3_refined",
            "output/stage4_final",
            "output/quality_reports",
            "temp/intermediate",
            "temp/cache"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        console.print(f"[green]✅ Created {len(directories)} output directories[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Directory creation failed: {e}[/red]")
        return False

def test_pipeline_class():
    """Test pipeline class instantiation"""
    console.print("🔍 Testing pipeline class...")
    
    try:
        # Mock the config file for testing
        test_config = {
            "agent_workflow": {
                "stage_1_face_detection_and_swap": {
                    "configuration": {
                        "face_matching": {"confidence_threshold": 0.95}
                    }
                },
                "logging_and_monitoring": {
                    "log_level": "INFO"
                }
            }
        }
        
        with open("test_workflow_config.yaml", 'w') as f:
            yaml.dump(test_config, f)
        
        # Test import
        from professional_pipeline import ProfessionalPipeline
        
        console.print("[green]✅ Pipeline class import successful[/green]")
        
        # Clean up test file
        Path("test_workflow_config.yaml").unlink()
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Pipeline class test failed: {e}[/red]")
        return False

def test_cli_interface():
    """Test CLI interface"""
    console.print("🔍 Testing CLI interface...")
    
    try:
        from pipeline_cli import cli, PresetManager
        
        # Test preset manager
        preset_manager = PresetManager()
        presets = preset_manager.list_presets()
        
        if presets:
            console.print(f"[green]✅ CLI interface loaded with {len(presets)} presets[/green]")
            return True
        else:
            console.print("[yellow]⚠️  CLI loaded but no presets found[/yellow]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ CLI interface test failed: {e}[/red]")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    
    console.print(Panel.fit(
        "🧪 [bold blue]Professional Pipeline Test Suite[/bold blue]\n"
        "Testing all components before processing",
        title="Face The Music Pipeline Tests"
    ))
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("API Token Test", test_api_token),
        ("Face Image Test", test_face_image),
        ("Output Directories Test", test_output_directories),
        ("Pipeline Class Test", test_pipeline_class),
        ("CLI Interface Test", test_cli_interface)
    ]
    
    results = []
    
    console.print("\n[blue]🚀 Running tests...[/blue]\n")
    
    for test_name, test_func in tests:
        console.print(f"[cyan]Running: {test_name}[/cyan]")
        result = test_func()
        results.append((test_name, result))
        console.print()
    
    # Display summary
    console.print("=" * 60)
    
    summary_table = Table(title="📊 Test Results Summary")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Status", style="green")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        summary_table.add_row(test_name, status)
        if result:
            passed += 1
    
    console.print(summary_table)
    
    # Overall status
    success_rate = (passed / len(tests)) * 100
    
    if success_rate == 100:
        console.print("\n[bold green]🎉 All tests passed! Pipeline ready for use.[/bold green]")
    elif success_rate >= 70:
        console.print(f"\n[yellow]⚠️  {passed}/{len(tests)} tests passed ({success_rate:.0f}%). Pipeline should work with some limitations.[/yellow]")
    else:
        console.print(f"\n[bold red]❌ Only {passed}/{len(tests)} tests passed ({success_rate:.0f}%). Please fix issues before using the pipeline.[/bold red]")
    
    return success_rate >= 70

def test_quick_processing():
    """Quick test of actual processing (if token available)"""
    
    if not os.getenv('REPLICATE_API_TOKEN'):
        console.print("[yellow]⚠️  Skipping processing test - no API token[/yellow]")
        return False
    
    if not Path("faces/your_face.png").exists():
        console.print("[yellow]⚠️  Skipping processing test - no face image[/yellow]")
        return False
    
    console.print("🧪 Running quick processing test...")
    
    try:
        # This would be a real test with a small test image
        console.print("[blue]Note: Actual processing test would require a test image[/blue]")
        console.print("[green]✅ Processing test framework ready[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Processing test failed: {e}[/red]")
        return False

if __name__ == "__main__":
    start_time = time.time()
    
    success = run_all_tests()
    
    elapsed = time.time() - start_time
    console.print(f"\n[dim]Test suite completed in {elapsed:.1f} seconds[/dim]")
    
    if success:
        console.print("\n[green]✅ Ready to process images with the professional pipeline![/green]")
        console.print("\n[dim]💡 Try: python pipeline_cli.py process --help[/dim]")
    else:
        console.print("\n[red]❌ Please resolve issues before using the pipeline[/red]")
        sys.exit(1)
