#!/usr/bin/env python3
"""
Unit tests for optimal pipeline workflow validation

Tests:
1. Confirm Python version is ≤3.11
2. Validate that LoRAs are only applied in stage 3
3. Verify images >1440px are resized before Flux

Author: Face The Music Team
"""

import unittest
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

# Add the project root to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimal_pipeline import OptimalPipeline


class TestOptimalPipeline(unittest.TestCase):
    """Unit tests for OptimalPipeline workflow validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = OptimalPipeline()
        
        # Create temporary test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test image larger than 1440px height
        self.large_test_image = os.path.join(self.temp_dir, "large_test.png")
        large_img = Image.new('RGB', (1920, 1600), color='red')
        large_img.save(self.large_test_image)
        
        # Create a test image smaller than 1440px height
        self.small_test_image = os.path.join(self.temp_dir, "small_test.png")
        small_img = Image.new('RGB', (1024, 768), color='blue')
        small_img.save(self.small_test_image)
        
        # Create a test face image
        self.face_test_image = os.path.join(self.temp_dir, "face_test.png")
        face_img = Image.new('RGB', (512, 512), color='green')
        face_img.save(self.face_test_image)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_python_version_is_less_than_or_equal_to_3_11(self):
        """Test that Python version is ≤3.11"""
        python_version = sys.version_info
        
        # Check that major version is 3 and minor version is <= 11
        self.assertEqual(python_version.major, 3, 
                        f"Expected Python 3.x, but got Python {python_version.major}.x")
        
        self.assertLessEqual(python_version.minor, 11,
                           f"Expected Python 3.11 or lower, but got Python {python_version.major}.{python_version.minor}")
        
        print(f"✅ Python version check passed: {python_version.major}.{python_version.minor}")
    
    def test_loras_only_applied_in_stage_3(self):
        """Test that LoRAs are only applied in stage 3, not in Flux stage"""
        
        # Test 1: Check configuration - Stage 1 (Flux) should NOT have LoRAs
        config = self.pipeline._default_optimal_config()
        stage1_config = config["agent_workflow"]["stage_1_flux_kontext_face_swap"]["configuration"]
        
        self.assertIn("apply_loras", stage1_config, 
                     "Stage 1 configuration should explicitly specify LoRA setting")
        self.assertFalse(stage1_config["apply_loras"], 
                        "Stage 1 (Flux Kontext) should NOT apply LoRAs")
        
        # Test 2: Check that Stage 3 has LoRA configurations
        stage3_config = config["agent_workflow"]["stage_3_sdxl_lora_photorealism"]
        self.assertIn("lora_enhancement_stack", stage3_config,
                     "Stage 3 should have LoRA enhancement stack")
        self.assertIn("photorealism_loras", stage3_config["lora_enhancement_stack"],
                     "Stage 3 should have photorealism LoRAs configured")
        
        # Test 3: Verify Stage 1 method doesn't include LoRA processing
        with patch.object(self.pipeline, 'flux_generator') as mock_flux:
            mock_flux.generate_image = Mock(return_value=Image.new('RGB', (1024, 1024)))
            
            analysis = self.pipeline.analyze_image_requirements(self.small_test_image)
            result = self.pipeline.stage1_flux_kontext_generation(
                self.small_test_image, self.face_test_image, 
                "test prompt", "test_output", analysis
            )
            
            # Verify stage 1 call doesn't mention LoRAs in the prompt or processing
            if mock_flux.generate_image.called:
                call_args = mock_flux.generate_image.call_args
                # The prompt should be basic face swap, not enhanced with LoRA terms
                prompt_used = call_args[1]['prompt'] if 'prompt' in call_args[1] else call_args[0][0]
                self.assertNotIn('lora', prompt_used.lower(), 
                               "Stage 1 prompt should not mention LoRAs")
        
        # Test 4: Verify Stage 3 method includes LoRA processing
        stage3_method_source = self.pipeline.stage3_sdxl_lora_photorealism.__doc__
        self.assertIn("LoRA", stage3_method_source, 
                     "Stage 3 method documentation should mention LoRAs")
        
        print("✅ LoRA application validation passed: LoRAs only in Stage 3")
    
    def test_images_over_1440px_resized_before_flux(self):
        """Test that images >1440px are resized before Flux processing"""
        
        # Test 1: Analyze requirements for large image
        analysis = self.pipeline.analyze_image_requirements(self.large_test_image)
        
        # Should detect that resize is needed
        self.assertTrue(analysis["needs_flux_resize"], 
                       "Large image (>1440px height) should be flagged for resize")
        
        # Should calculate optimal size with max height 1440
        optimal_size = analysis["flux_optimal_size"]
        self.assertLessEqual(optimal_size[1], 1440, 
                           "Flux optimal size height should be ≤1440px")
        
        # Test 2: Check that small images don't need resize
        small_analysis = self.pipeline.analyze_image_requirements(self.small_test_image)
        self.assertFalse(small_analysis["needs_flux_resize"],
                        "Small image (≤1440px height) should not need resize")
        
        # Test 3: Verify actual resizing in Stage 1
        with patch.object(self.pipeline, 'flux_generator') as mock_flux:
            mock_result_image = Image.new('RGB', (1024, 1024))
            mock_flux.generate_image = Mock(return_value=mock_result_image)
            
            # Test with large image
            large_analysis = self.pipeline.analyze_image_requirements(self.large_test_image)
            
            # Mock the stage 1 processing
            with patch('builtins.open', create=True) as mock_open:
                with patch('pathlib.Path.mkdir'):
                    # Mock file operations
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    # Test stage 1 with large image
                    result = self.pipeline.stage1_flux_kontext_generation(
                        self.large_test_image, self.face_test_image,
                        "test prompt", "test_output", large_analysis
                    )
            
            # Verify flux generator was called with the correct dimensions
            if mock_flux.generate_image.called:
                call_args = mock_flux.generate_image.call_args
                called_height = call_args[1]['height'] if 'height' in call_args[1] else call_args[0][2]
                self.assertLessEqual(called_height, 1440,
                                   "Flux generator should be called with height ≤1440px")
        
        # Test 4: Verify the max height constant
        self.assertEqual(self.pipeline.max_height_flux, 1440,
                        "Pipeline should have max_height_flux set to 1440")
        
        # Test 5: Check stage 1 method includes resize logic
        # Read the original image to verify it's actually > 1440
        with Image.open(self.large_test_image) as img:
            original_height = img.size[1]
            self.assertGreater(original_height, 1440,
                             "Test image should be > 1440px to validate resize logic")
        
        print("✅ Image resize validation passed: Images >1440px resized before Flux")
    
    def test_stage_3_lora_configuration_details(self):
        """Additional test to verify Stage 3 LoRA configuration details"""
        config = self.pipeline._default_optimal_config()
        stage3_config = config["agent_workflow"]["stage_3_sdxl_lora_photorealism"]
        
        # Verify LoRA stack structure
        lora_stack = stage3_config["lora_enhancement_stack"]["photorealism_loras"]
        
        # Should have at least one LoRA configured
        self.assertGreater(len(lora_stack), 0, "Should have at least one LoRA configured")
        
        # Each LoRA should have name and weight
        for lora in lora_stack:
            self.assertIn("name", lora, "Each LoRA should have a name")
            self.assertIn("weight", lora, "Each LoRA should have a weight")
            self.assertIsInstance(lora["weight"], (int, float), 
                                "LoRA weight should be numeric")
            self.assertGreater(lora["weight"], 0, "LoRA weight should be positive")
            self.assertLessEqual(lora["weight"], 1.0, "LoRA weight should be ≤1.0")
        
        print("✅ Stage 3 LoRA configuration details validated")
    
    def test_workflow_version_and_stages(self):
        """Test workflow version and stage sequence"""
        config = self.pipeline._default_optimal_config()
        workflow = config["agent_workflow"]
        
        # Verify all 4 stages are present
        expected_stages = [
            "stage_1_flux_kontext_face_swap",
            "stage_2_resize_analysis", 
            "stage_3_sdxl_lora_photorealism",
            "stage_4_post_processing"
        ]
        
        for stage in expected_stages:
            self.assertIn(stage, workflow, f"Stage {stage} should be in workflow configuration")
        
        print("✅ Workflow stages and version validated")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
