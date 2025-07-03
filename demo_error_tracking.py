#!/usr/bin/env python3
"""
Demo: Error Tracking Integration
Shows how the error tracking system works with the pipeline.
"""

import os
from pathlib import Path
from error_tracker import track_errors, error_tracking_context, _error_tracker
from replicate_generator import ReplicateFluxGenerator
from optimal_pipeline import OptimalPipeline

def demo_error_tracking():
    """Demonstrate error tracking capabilities."""
    print("🔧 Demo: Error Tracking System")
    print("=" * 40)
    
    # Show current status
    print("\n📊 Current Error Status:")
    summary = _error_tracker.get_error_summary()
    print(f"Active Errors: {summary['active_errors_count']}")
    print(f"Fixed Errors: {summary['fixed_errors_count']}")
    print(f"Total Tracked: {summary['total_errors_tracked']}")
    
    # Demo 1: Function with error tracking decorator
    print("\n🎯 Demo 1: Function with Error Tracking Decorator")
    
    @track_errors(context="demo_function", severity="HIGH")
    def problematic_function():
        """A function that will throw an error for demo purposes."""
        raise ValueError("This is a demo error to show automatic tracking")
    
    try:
        problematic_function()
    except ValueError as e:
        print(f"⚠️  Caught expected error: {e}")
    
    # Demo 2: Context manager for error tracking
    print("\n🎯 Demo 2: Context Manager Error Tracking")
    
    try:
        with error_tracking_context("demo_context_block", "MEDIUM"):
            # Simulate some processing that fails
            if not os.path.exists("/nonexistent/path/image.jpg"):
                raise FileNotFoundError("Demo file not found error")
    except FileNotFoundError as e:
        print(f"⚠️  Caught expected error: {e}")
    
    # Show updated status
    print("\n📊 Updated Error Status:")
    summary = _error_tracker.get_error_summary()
    print(f"Active Errors: {summary['active_errors_count']}")
    print(f"Fixed Errors: {summary['fixed_errors_count']}")
    print(f"Total Tracked: {summary['total_errors_tracked']}")
    
    # Show active errors
    print("\n📋 Active Errors:")
    errors = _error_tracker.get_active_errors()
    for error_hash, error_info in errors.items():
        print(f"🔴 {error_hash}: {error_info['error_type']}")
        print(f"   Message: {error_info['error_message']}")
        print(f"   Location: {error_info['location']}")
        print(f"   Severity: {error_info['severity']}")
        print(f"   Occurrences: {error_info['occurrences']}")
        print()

def demo_pipeline_integration():
    """Show how error tracking integrates with the pipeline."""
    print("\n🚀 Demo: Pipeline Integration")
    print("=" * 40)
    
    # Try to initialize pipeline components
    try:
        # This should work
        pipeline = OptimalPipeline()
        print("✅ OptimalPipeline initialized successfully")
        
        # This might fail if no API key
        if os.getenv('REPLICATE_API_TOKEN'):
            generator = ReplicateFluxGenerator()
            print("✅ ReplicateFluxGenerator initialized successfully")
        else:
            print("⚠️  Replicate API token not found - some features will be limited")
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
    
    print("\n💡 How to use the error tracking:")
    print("1. Errors are automatically tracked when they occur in decorated functions")
    print("2. Check error status: python3 error_tracker.py status")
    print("3. List active errors: python3 error_tracker.py list")
    print("4. Mark errors as fixed: python3 error_tracker.py fix <error_hash> -d 'Description of fix'")
    print("5. Fixed errors are automatically moved to FIXED_ERRORS.md")

def main():
    """Main demo function."""
    demo_error_tracking()
    demo_pipeline_integration()
    
    print("\n🎉 Demo Complete!")
    print("\nNext steps:")
    print("- Run your pipeline normally - errors will be automatically tracked")
    print("- Check TODO.md for auto-added error tasks")
    print("- Use the CLI to manage errors: python3 error_tracker.py status")
    print("- Review FIXED_ERRORS.md for resolved issues")

if __name__ == "__main__":
    main()
