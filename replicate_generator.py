import os
import replicate
import requests
from PIL import Image
from io import BytesIO
import time
from typing import Optional, Dict, Any
from error_tracker import track_errors, error_tracking_context


class ReplicateFluxGenerator:
    """
    Replicate Flux image generator with face swap and upscaling capabilities.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the Replicate Flux generator."""
        self.api_key = api_key or os.getenv('REPLICATE_API_TOKEN') or os.getenv('REPLICATE_API_KEY')
        if not self.api_key:
            raise ValueError("Replicate API key is required. Set REPLICATE_API_TOKEN environment variable.")
        
        # Set the API key for replicate
        os.environ['REPLICATE_API_TOKEN'] = self.api_key
        
        # Flux model versions
        self.flux_models = {
            'flux-kontext-pro': 'black-forest-labs/flux-kontext-pro',
            'flux-1.1-pro': 'black-forest-labs/flux-1.1-pro',
            'flux-pro': 'black-forest-labs/flux-pro',
            'flux-dev': 'black-forest-labs/flux-dev',
            'flux-schnell': 'black-forest-labs/flux-schnell'
        }
        
        # Upscaler models
        self.upscaler_models = {
            'real-esrgan': 'philz1337x/clarity-upscaler',
            'esrgan-x4': 'mv-lab/swin2sr',
            'upscaler': 'tencentarc/gfpgan'
        }
        
    @track_errors(context="flux_image_generation", severity="HIGH")
    def generate_image(self, 
                      prompt: str, 
                      negative_prompt: str = "", 
                      width: int = 1024, 
                      height: int = 1024,
                      steps: int = 4,
                      guidance_scale: float = 0.0,
                      seed: int = None,
                      model: str = 'flux-schnell',
                      reference_image_path: str = None) -> Optional[Image.Image]:
        """
        Generate an image using Replicate Flux.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (not used in Flux Schnell)
            width: Image width
            height: Image height  
            steps: Number of inference steps
            guidance_scale: Guidance scale (0.0 for Flux Schnell)
            seed: Random seed
            model: Flux model to use
            
        Returns:
            PIL Image or None if failed
        """
        try:
            print(f"🎨 Generating image with Flux {model}...")
            print(f"📝 Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Prompt: {prompt}")
            
            # Get the model
            model_name = self.flux_models.get(model, self.flux_models['flux-schnell'])
            
            # Prepare input parameters
            input_params = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "output_format": "png",
                "output_quality": 95
            }
            
            # Add face reference for Kontext Pro
            if model == 'flux-kontext-pro' and reference_image_path and os.path.exists(reference_image_path):
                print(f"👤 Using face reference: {reference_image_path}")
                # Kontext Pro expects input_image as a file path/URI
                input_params["input_image"] = open(reference_image_path, 'rb')
            
            # Note: Flux Kontext Pro doesn't support LoRAs
            # Photorealism achieved through native model quality and optimized prompts
            # LoRAs should only be used in Stage 3 (SDXL) for photorealism enhancement
            
            # Add seed if provided
            if seed and seed != -1:
                input_params["seed"] = seed
                
            # Adjust parameters based on model type
            if model in ['flux-dev', 'flux-1.1-pro', 'flux-pro']:
                if negative_prompt:
                    input_params["negative_prompt"] = negative_prompt
                if guidance_scale == 0.0:  # Use default guidance for pro models
                    input_params["guidance_scale"] = 3.5
                # Pro models can handle more steps
                if model == 'flux-1.1-pro' and steps == 4:
                    input_params["num_inference_steps"] = 25
                    
            print(f"⚙️ Model: {model_name}")
            print(f"⚙️ Settings: {width}x{height}, {steps} steps, guidance: {guidance_scale}")
            
            # Run the model
            output = replicate.run(model_name, input=input_params)
            
            if output:
                # Handle different output formats
                if isinstance(output, list) and len(output) > 0:
                    image_url = output[0]
                elif isinstance(output, str):
                    image_url = output
                elif hasattr(output, 'url'):  # Handle FileOutput from Flux 1.1 Pro
                    image_url = output.url
                elif hasattr(output, 'read'):  # Handle file-like objects
                    # For direct file content, convert to PIL Image
                    try:
                        image = Image.open(output)
                        print(f"✅ Image generated successfully: {image.size}")
                        return image
                    except Exception as e:
                        print(f"❌ Error reading file output: {e}")
                        return None
                else:
                    print(f"❌ Unexpected output format: {type(output)}")
                    print(f"❌ Output attributes: {dir(output)}")
                    return None
                
                print(f"📥 Downloading image from: {image_url}")
                
                # Download the image
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Convert to PIL Image
                image = Image.open(BytesIO(response.content))
                
                print(f"✅ Image generated successfully: {image.size}")
                return image
                
            else:
                print("❌ No output received from Replicate")
                return None
                
        except Exception as e:
            print(f"❌ Error generating image with Flux: {e}")
            return None
    
    @track_errors(context="image_upscaling", severity="MEDIUM")
    def upscale_image(self, 
                     image: Image.Image, 
                     scale: int = 2, 
                     model: str = 'real-esrgan') -> Optional[Image.Image]:
        """
        Upscale an image using Replicate upscaling models or local fallback.
        
        Args:
            image: PIL Image to upscale
            scale: Upscaling factor (2 or 4)
            model: Upscaling model to use
            
        Returns:
            Upscaled PIL Image or None if failed
        """
        # Try local upscaling first as it's more reliable
        try:
            print(f"🔍 Upscaling image locally with PIL Lanczos (scale: {scale}x)...")
            
            original_size = image.size
            new_width = original_size[0] * scale
            new_height = original_size[1] * scale
            
            # Use Lanczos resampling for high-quality upscaling
            upscaled_image = image.resize(
                (new_width, new_height), 
                Image.Resampling.LANCZOS
            )
            
            print(f"✅ Image upscaled successfully: {original_size} → {upscaled_image.size}")
            return upscaled_image
            
        except Exception as e:
            print(f"❌ Local upscaling failed: {e}")
            
        # Fallback to Replicate models if local fails
        try:
            print(f"🔍 Trying Replicate upscaling with {model} (scale: {scale}x)...")
            
            # Save image to bytes for upload
            img_bytes = BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Get the upscaler model
            model_name = self.upscaler_models.get(model, self.upscaler_models['real-esrgan'])
            
            # Prepare input
            input_params = {
                "image": img_bytes,
                "scale": scale
            }
            
            print(f"⚙️ Upscaler: {model_name}")
            
            # Run upscaling
            output = replicate.run(model_name, input=input_params)
            
            if output:
                # Handle output
                if isinstance(output, list) and len(output) > 0:
                    image_url = output[0]
                elif isinstance(output, str):
                    image_url = output
                else:
                    print(f"❌ Unexpected upscaler output format: {type(output)}")
                    return image  # Return original instead of None
                
                print(f"📥 Downloading upscaled image from: {image_url}")
                
                # Download the upscaled image
                response = requests.get(image_url, timeout=60)
                response.raise_for_status()
                
                # Convert to PIL Image
                upscaled_image = Image.open(BytesIO(response.content))
                
                print(f"✅ Replicate upscaling successful: {image.size} → {upscaled_image.size}")
                return upscaled_image
                
            else:
                print("❌ No output received from Replicate upscaler")
                return image  # Return original instead of None
                
        except Exception as e:
            print(f"❌ Replicate upscaling failed: {e}")
            print("⚠️ Returning original image")
            return image
    
    @track_errors(context="flux_workflow", severity="HIGH")
    def generate_with_face_swap_and_upscale(self,
                                          prompt: str,
                                          face_source_path: str,
                                          config: Dict[str, Any],
                                          sd_settings: Dict[str, Any],
                                          enhance: bool = False,
                                          upscale: bool = False) -> Optional[Image.Image]:
        """
        Complete workflow: Generate image with Flux → Face swap → Upscale
        
        Args:
            prompt: Text prompt for generation
            face_source_path: Path to source face image
            config: Configuration dictionary
            sd_settings: SD settings dictionary
            enhance: Whether to enhance (not implemented yet)
            upscale: Whether to upscale
            
        Returns:
            Final processed PIL Image or None if failed
        """
        try:
            # Step 1: Generate base image with Flux
            negative_prompt = sd_settings.get('negative_prompt', '')
            width = sd_settings.get('width', 1024)
            height = sd_settings.get('height', 1024)
            steps = sd_settings.get('steps', 4)
            guidance_scale = sd_settings.get('cfg', 0.0)
            seed = sd_settings.get('seed', None)
            
            # Use flux-kontext-pro for native face swapping
            flux_model = config.get('flux', {}).get('model', 'flux-kontext-pro')
            
            # Note: LoRAs are NOT used in Flux generation (Stage 1)
            # LoRAs are applied later in Stage 3 (SDXL) for photorealism enhancement
            
            # Use face reference if available and using Kontext Pro
            reference_image = None
            if flux_model == 'flux-kontext-pro' and face_source_path:
                reference_image = face_source_path
            
            generated_image = self.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                model=flux_model,
                reference_image_path=reference_image
            )
            
            if not generated_image:
                print("❌ Failed to generate base image")
                return None
            
            # Face swapping is now handled natively by Flux Kontext Pro
            # No additional face swap processing needed
            
            # Step 3: Upscale if requested
            if upscale:
                upscale_config = config.get('upscale', {})
                upscale_model = upscale_config.get('model', 'real-esrgan')
                upscale_factor = upscale_config.get('scale', 2)
                
                upscaled_image = self.upscale_image(
                    image=generated_image,
                    scale=upscale_factor,
                    model=upscale_model
                )
                
                if upscaled_image:
                    generated_image = upscaled_image
            
            print("🎉 Complete workflow finished successfully!")
            return generated_image
            
        except Exception as e:
            print(f"❌ Error in complete workflow: {e}")
            return None


def test_replicate_flux():
    """Test function for Replicate Flux generation."""
    try:
        generator = ReplicateFluxGenerator()
        
        # Test prompt
        prompt = "A beautiful landscape with mountains and a lake at sunset, photorealistic, 8k"
        
        # Generate test image
        image = generator.generate_image(
            prompt=prompt,
            width=1024,
            height=1024,
            steps=4,
            model='flux-schnell'
        )
        
        if image:
            # Save test image
            output_path = "output/flux_test.png"
            os.makedirs("output", exist_ok=True)
            image.save(output_path)
            print(f"✅ Test image saved to: {output_path}")
            return True
        else:
            print("❌ Test failed: No image generated")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


if __name__ == "__main__":
    test_replicate_flux()
