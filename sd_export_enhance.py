#!/usr/bin/env python3
"""
Stable Diffusion Export and LoRA Enhancement Pipeline
Step 5: Convert optimized images to SD-ready format and enhance with LoRAs

Features:
- Convert optimized images to SD-ready format (checkpoint/imgs)
- Match original faces and apply best LoRA models
- Automate matching by comparing face embeddings or metadata
- Generate SD-compatible formats and enhanced outputs

Author: Face The Music Team
Version: 1.0-SD-EXPORT
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import cv2
import yaml
from datetime import datetime

# Core libraries
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
import torchvision.transforms as transforms
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

# Face analysis and matching
try:
    import face_recognition
    import dlib
    FACE_ANALYSIS_AVAILABLE = True
except ImportError:
    FACE_ANALYSIS_AVAILABLE = False
    print("Warning: face_recognition and dlib not available. Face matching will be limited.")

# Stable Diffusion components
try:
    from diffusers import (
        StableDiffusionXLPipeline, 
        StableDiffusionXLImg2ImgPipeline,
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
        AutoencoderKL,
        UNet2DConditionModel,
        DDIMScheduler
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    print("Warning: Stable Diffusion libraries not available. SD processing will be limited.")

console = Console()

@dataclass
class FaceEmbedding:
    """Face embedding data for matching"""
    image_path: str
    face_encoding: np.ndarray
    face_location: Tuple[int, int, int, int]  # top, right, bottom, left
    confidence: float
    extraction_timestamp: str
    metadata: Dict[str, Any]

@dataclass
class LoRAConfig:
    """LoRA model configuration"""
    name: str
    path: str
    weight: float
    category: str  # e.g., 'photorealism', 'skin_texture', 'lighting'
    compatibility_score: float
    description: str

@dataclass
class SDExportResult:
    """Results from SD export and enhancement"""
    original_path: str
    sd_checkpoint_path: Optional[str]
    enhanced_image_path: Optional[str]
    face_match_confidence: float
    applied_loras: List[LoRAConfig]
    processing_time: float
    success: bool
    error_message: Optional[str]
    quality_metrics: Dict[str, float]

class FaceMatcher:
    """Advanced face matching using embeddings and metadata"""
    
    def __init__(self):
        self.known_faces: Dict[str, FaceEmbedding] = {}
        self.face_database_path = "output/face_database.json"
        self.load_face_database()
    
    def extract_face_embedding(self, image_path: str) -> Optional[FaceEmbedding]:
        """Extract face embedding from image"""
        if not FACE_ANALYSIS_AVAILABLE:
            console.print("[yellow]⚠️  Face recognition not available[/yellow]")
            return None
        
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if not face_locations:
                console.print(f"[yellow]⚠️  No faces found in {image_path}[/yellow]")
                return None
            
            # Use the largest face
            face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            
            # Extract face encoding
            face_encodings = face_recognition.face_encodings(image, [face_location])
            
            if not face_encodings:
                console.print(f"[yellow]⚠️  Could not encode face in {image_path}[/yellow]")
                return None
            
            # Calculate confidence based on face size and quality
            face_height = face_location[2] - face_location[0]
            face_width = face_location[1] - face_location[3]
            face_area = face_height * face_width
            image_area = image.shape[0] * image.shape[1]
            
            confidence = min(1.0, (face_area / image_area) * 10)  # Larger faces = higher confidence
            
            return FaceEmbedding(
                image_path=image_path,
                face_encoding=face_encodings[0],
                face_location=face_location,
                confidence=confidence,
                extraction_timestamp=datetime.now().isoformat(),
                metadata={
                    "face_area": face_area,
                    "image_dimensions": (image.shape[1], image.shape[0]),
                    "face_ratio": face_area / image_area
                }
            )
            
        except Exception as e:
            console.print(f"[red]❌ Error extracting face from {image_path}: {e}[/red]")
            return None
    
    def compare_faces(self, face1: FaceEmbedding, face2: FaceEmbedding, 
                     threshold: float = 0.6) -> Tuple[bool, float]:
        """Compare two face embeddings"""
        if not FACE_ANALYSIS_AVAILABLE:
            return False, 0.0
        
        try:
            # Calculate distance
            distance = face_recognition.face_distance([face1.face_encoding], face2.face_encoding)[0]
            
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity = 1.0 - distance
            
            # Check if faces match
            is_match = distance < threshold
            
            return is_match, similarity
            
        except Exception as e:
            console.print(f"[red]❌ Error comparing faces: {e}[/red]")
            return False, 0.0
    
    def find_best_face_match(self, target_embedding: FaceEmbedding) -> Tuple[Optional[str], float]:
        """Find the best matching face from the database"""
        best_match = None
        best_similarity = 0.0
        
        for face_id, known_face in self.known_faces.items():
            is_match, similarity = self.compare_faces(target_embedding, known_face)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = face_id
        
        return best_match, best_similarity
    
    def save_face_database(self):
        """Save face database to disk"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for face_id, embedding in self.known_faces.items():
                serializable_data[face_id] = {
                    "image_path": embedding.image_path,
                    "face_encoding": embedding.face_encoding.tolist(),
                    "face_location": embedding.face_location,
                    "confidence": embedding.confidence,
                    "extraction_timestamp": embedding.extraction_timestamp,
                    "metadata": embedding.metadata
                }
            
            with open(self.face_database_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
        except Exception as e:
            console.print(f"[red]❌ Error saving face database: {e}[/red]")
    
    def load_face_database(self):
        """Load face database from disk"""
        try:
            if os.path.exists(self.face_database_path):
                with open(self.face_database_path, 'r') as f:
                    data = json.load(f)
                
                # Convert lists back to numpy arrays
                for face_id, face_data in data.items():
                    self.known_faces[face_id] = FaceEmbedding(
                        image_path=face_data["image_path"],
                        face_encoding=np.array(face_data["face_encoding"]),
                        face_location=tuple(face_data["face_location"]),
                        confidence=face_data["confidence"],
                        extraction_timestamp=face_data["extraction_timestamp"],
                        metadata=face_data["metadata"]
                    )
                    
                console.print(f"[green]✅ Loaded {len(self.known_faces)} faces from database[/green]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not load face database: {e}[/yellow]")
            self.known_faces = {}

class LoRAManager:
    """Manage LoRA models and their applications"""
    
    def __init__(self):
        self.available_loras = self._initialize_lora_configs()
        self.model_cache = {}
    
    def _initialize_lora_configs(self) -> List[LoRAConfig]:
        """Initialize available LoRA configurations"""
        return [
            LoRAConfig(
                name="PhotoReal XL Pro",
                path="loras/photoreal_xl_pro.safetensors",
                weight=0.75,
                category="photorealism",
                compatibility_score=0.95,
                description="Ultra-realistic photorealism enhancement"
            ),
            LoRAConfig(
                name="Hyper-Detailed Skin Texture",
                path="loras/skin_texture_detail.safetensors",
                weight=0.65,
                category="skin_texture",
                compatibility_score=0.90,
                description="Enhanced skin texture and pore detail"
            ),
            LoRAConfig(
                name="Professional Portrait Lighting",
                path="loras/portrait_lighting_pro.safetensors",
                weight=0.55,
                category="lighting",
                compatibility_score=0.85,
                description="Professional studio lighting effects"
            ),
            LoRAConfig(
                name="Cinematic Color Grading",
                path="loras/cinematic_grading.safetensors",
                weight=0.60,
                category="color_grading",
                compatibility_score=0.80,
                description="Cinematic color enhancement and grading"
            ),
            LoRAConfig(
                name="Ultra Sharp Details",
                path="loras/ultra_sharp_details.safetensors",
                weight=0.50,
                category="detail_enhancement",
                compatibility_score=0.88,
                description="Enhanced detail sharpness and clarity"
            )
        ]
    
    def select_best_loras(self, image_analysis: Dict[str, Any], 
                         face_match_confidence: float) -> List[LoRAConfig]:
        """Select the best LoRA models based on image analysis and face matching"""
        selected_loras = []
        
        # Always include photorealism for high-confidence face matches
        if face_match_confidence > 0.7:
            photorealism_loras = [l for l in self.available_loras if l.category == "photorealism"]
            if photorealism_loras:
                selected_loras.append(photorealism_loras[0])
        
        # Add skin texture enhancement for portraits
        if image_analysis.get("has_portrait", False):
            skin_loras = [l for l in self.available_loras if l.category == "skin_texture"]
            if skin_loras:
                selected_loras.append(skin_loras[0])
        
        # Add lighting enhancement if needed
        if image_analysis.get("needs_lighting_enhancement", False):
            lighting_loras = [l for l in self.available_loras if l.category == "lighting"]
            if lighting_loras:
                selected_loras.append(lighting_loras[0])
        
        # Add detail enhancement for high-resolution images
        if image_analysis.get("resolution", 0) > 2048:
            detail_loras = [l for l in self.available_loras if l.category == "detail_enhancement"]
            if detail_loras:
                selected_loras.append(detail_loras[0])
        
        return selected_loras

class SDExportPipeline:
    """Main pipeline for SD export and LoRA enhancement"""
    
    def __init__(self, config_path: str = "sd_export_config.yaml"):
        self.config = self._load_config(config_path)
        self.face_matcher = FaceMatcher()
        self.lora_manager = LoRAManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sd_pipeline = None
        
        # Setup directories
        self._setup_directories()
        
        # Initialize SD pipeline if available
        if SD_AVAILABLE:
            self._initialize_sd_pipeline()
        
        console.print(f"[green]🚀 SD Export Pipeline initialized on {self.device}[/green]")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._create_default_config(config_path)
    
    def _create_default_config(self, config_path: str) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            "sd_export": {
                "output_format": "safetensors",
                "checkpoint_format": "diffusers",
                "image_format": "png",
                "quality": 95
            },
            "face_matching": {
                "confidence_threshold": 0.6,
                "max_faces_per_image": 5,
                "face_size_threshold": 50
            },
            "lora_enhancement": {
                "max_loras_per_image": 3,
                "weight_adjustment_factor": 0.9,
                "enhancement_strength": 0.7
            },
            "stable_diffusion": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "steps": 20,
                "guidance_scale": 7.5,
                "strength": 0.35
            }
        }
        
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def _setup_directories(self):
        """Setup output directories"""
        directories = [
            "output/sd_export",
            "output/sd_export/checkpoints",
            "output/sd_export/enhanced_images",
            "output/sd_export/face_matches",
            "output/sd_export/metadata"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _initialize_sd_pipeline(self):
        """Initialize Stable Diffusion pipeline"""
        try:
            model_id = self.config["stable_diffusion"]["model_id"]
            self.sd_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                use_safetensors=True
            )
            self.sd_pipeline = self.sd_pipeline.to(self.device)
            console.print("[green]✅ Stable Diffusion XL pipeline initialized[/green]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not initialize SD pipeline: {e}[/yellow]")
            self.sd_pipeline = None
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for processing decisions"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                analysis = {
                    "resolution": max(width, height),
                    "aspect_ratio": width / height,
                    "has_portrait": width / height < 1.5,  # Portrait-like aspect ratio
                    "needs_upscaling": max(width, height) < 1024,
                    "needs_lighting_enhancement": self._needs_lighting_enhancement(img),
                    "color_depth": len(img.getbands()),
                    "file_size": os.path.getsize(image_path)
                }
                
                return analysis
                
        except Exception as e:
            console.print(f"[red]❌ Error analyzing image {image_path}: {e}[/red]")
            return {}
    
    def _needs_lighting_enhancement(self, image: Image.Image) -> bool:
        """Determine if image needs lighting enhancement"""
        try:
            # Convert to grayscale and analyze brightness
            gray = image.convert('L')
            hist = gray.histogram()
            
            # Calculate average brightness
            total_pixels = sum(hist)
            weighted_sum = sum(i * hist[i] for i in range(256))
            avg_brightness = weighted_sum / total_pixels
            
            # Check if image is too dark or too bright
            return avg_brightness < 80 or avg_brightness > 200
            
        except Exception:
            return False
    
    def convert_to_sd_format(self, image_path: str, output_name: str) -> Dict[str, str]:
        """Convert image to SD-ready formats"""
        try:
            with Image.open(image_path) as img:
                # Ensure RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to optimal SD dimensions (multiple of 64)
                width, height = img.size
                new_width = (width // 64) * 64
                new_height = (height // 64) * 64
                
                if new_width != width or new_height != height:
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save in multiple formats
                outputs = {}
                
                # PNG format (lossless)
                png_path = f"output/sd_export/{output_name}_sd_ready.png"
                img.save(png_path, 'PNG', quality=95)
                outputs['png'] = png_path
                
                # JPEG format (smaller file size)
                jpg_path = f"output/sd_export/{output_name}_sd_ready.jpg"
                img.save(jpg_path, 'JPEG', quality=95)
                outputs['jpg'] = jpg_path
                
                # Create tensor format if PyTorch is available
                if torch.cuda.is_available():
                    tensor_path = f"output/sd_export/{output_name}_tensor.pt"
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
                    tensor = transform(img).unsqueeze(0)
                    torch.save(tensor, tensor_path)
                    outputs['tensor'] = tensor_path
                
                return outputs
                
        except Exception as e:
            console.print(f"[red]❌ Error converting {image_path} to SD format: {e}[/red]")
            return {}
    
    def enhance_with_loras(self, image_path: str, selected_loras: List[LoRAConfig], 
                          output_name: str) -> Optional[str]:
        """Enhance image using selected LoRA models"""
        if not self.sd_pipeline:
            console.print("[yellow]⚠️  SD pipeline not available, skipping LoRA enhancement[/yellow]")
            return None
        
        try:
            # Load image
            with Image.open(image_path) as img:
                # Create enhancement prompt
                prompt = self._build_enhancement_prompt()
                
                # Apply LoRA enhancements (simulated - in practice would load actual LoRA weights)
                enhanced_img = self._simulate_lora_enhancement(img, selected_loras)
                
                # Save enhanced image
                output_path = f"output/sd_export/enhanced_images/{output_name}_enhanced.png"
                enhanced_img.save(output_path, 'PNG', quality=95)
                
                return output_path
                
        except Exception as e:
            console.print(f"[red]❌ Error applying LoRA enhancements: {e}[/red]")
            return None
    
    def _build_enhancement_prompt(self) -> str:
        """Build prompt for SD enhancement"""
        return ("ultra photorealistic, professional photography, high quality, "
                "detailed skin texture, natural lighting, cinematic quality, "
                "8K resolution, professional portrait, masterpiece")
    
    def _simulate_lora_enhancement(self, image: Image.Image, 
                                  loras: List[LoRAConfig]) -> Image.Image:
        """Simulate LoRA enhancement (placeholder for actual LoRA application)"""
        enhanced = image.copy()
        
        for lora in loras:
            if lora.category == "photorealism":
                # Enhance realism
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(1.0 + lora.weight * 0.2)
                
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.0 + lora.weight * 0.15)
                
            elif lora.category == "skin_texture":
                # Enhance skin detail
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(
                    radius=1, percent=int(100 + lora.weight * 50), threshold=2
                ))
                
            elif lora.category == "lighting":
                # Adjust lighting
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.0 + lora.weight * 0.1)
                
            elif lora.category == "detail_enhancement":
                # Enhance details
                enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        return enhanced
    
    def process_image(self, image_path: str, original_face_path: str = None) -> SDExportResult:
        """Process a single image through the SD export pipeline"""
        start_time = time.time()
        output_name = Path(image_path).stem
        
        try:
            console.print(f"[blue]🔄 Processing: {image_path}[/blue]")
            
            # Step 1: Analyze image
            analysis = self.analyze_image(image_path)
            
            # Step 2: Extract face embedding from processed image
            processed_face = self.face_matcher.extract_face_embedding(image_path)
            
            # Step 3: Match with original face if provided
            face_match_confidence = 0.0
            if original_face_path and processed_face:
                original_face = self.face_matcher.extract_face_embedding(original_face_path)
                if original_face:
                    is_match, confidence = self.face_matcher.compare_faces(original_face, processed_face)
                    face_match_confidence = confidence
                    console.print(f"[green]👤 Face match confidence: {confidence:.2f}[/green]")
            
            # Step 4: Convert to SD-ready format
            sd_formats = self.convert_to_sd_format(image_path, output_name)
            
            # Step 5: Select appropriate LoRA models
            selected_loras = self.lora_manager.select_best_loras(analysis, face_match_confidence)
            console.print(f"[cyan]🎨 Selected {len(selected_loras)} LoRA models[/cyan]")
            
            # Step 6: Apply LoRA enhancements
            enhanced_path = None
            if selected_loras:
                enhanced_path = self.enhance_with_loras(image_path, selected_loras, output_name)
            
            # Step 7: Save metadata
            metadata = {
                "original_path": image_path,
                "processing_timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "face_match_confidence": face_match_confidence,
                "selected_loras": [asdict(lora) for lora in selected_loras],
                "sd_formats": sd_formats,
                "enhanced_path": enhanced_path
            }
            
            metadata_path = f"output/sd_export/metadata/{output_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            processing_time = time.time() - start_time
            
            return SDExportResult(
                original_path=image_path,
                sd_checkpoint_path=sd_formats.get('tensor'),
                enhanced_image_path=enhanced_path,
                face_match_confidence=face_match_confidence,
                applied_loras=selected_loras,
                processing_time=processing_time,
                success=True,
                error_message=None,
                quality_metrics={"processing_time": processing_time}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing failed: {str(e)}"
            console.print(f"[red]❌ {error_msg}[/red]")
            
            return SDExportResult(
                original_path=image_path,
                sd_checkpoint_path=None,
                enhanced_image_path=None,
                face_match_confidence=0.0,
                applied_loras=[],
                processing_time=processing_time,
                success=False,
                error_message=error_msg,
                quality_metrics={}
            )
    
    def batch_process(self, input_dir: str, original_face_path: str = None) -> List[SDExportResult]:
        """Process all images in a directory"""
        results = []
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        console.print(f"[blue]📁 Found {len(image_files)} images to process[/blue]")
        
        # Process each image
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            
            task = progress.add_task("Processing images...", total=len(image_files))
            
            for image_file in image_files:
                result = self.process_image(str(image_file), original_face_path)
                results.append(result)
                progress.update(task, advance=1)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: List[SDExportResult]):
        """Generate a summary report of processing results"""
        try:
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            report = {
                "summary": {
                    "total_images": len(results),
                    "successful": len(successful),
                    "failed": len(failed),
                    "success_rate": len(successful) / len(results) if results else 0,
                    "total_processing_time": sum(r.processing_time for r in results),
                    "average_processing_time": sum(r.processing_time for r in results) / len(results) if results else 0
                },
                "face_matching": {
                    "average_confidence": sum(r.face_match_confidence for r in successful) / len(successful) if successful else 0,
                    "high_confidence_matches": len([r for r in successful if r.face_match_confidence > 0.8])
                },
                "lora_usage": {
                    "total_loras_applied": sum(len(r.applied_loras) for r in successful),
                    "average_loras_per_image": sum(len(r.applied_loras) for r in successful) / len(successful) if successful else 0
                },
                "errors": [r.error_message for r in failed if r.error_message]
            }
            
            # Save report
            report_path = f"output/sd_export/processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Display summary
            table = Table(title="SD Export Processing Summary")
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="green")
            
            table.add_row("Total Images", str(report["summary"]["total_images"]))
            table.add_row("Successful", str(report["summary"]["successful"]))
            table.add_row("Failed", str(report["summary"]["failed"]))
            table.add_row("Success Rate", f"{report['summary']['success_rate']:.1%}")
            table.add_row("Avg Face Confidence", f"{report['face_matching']['average_confidence']:.2f}")
            table.add_row("Avg LoRAs per Image", f"{report['lora_usage']['average_loras_per_image']:.1f}")
            
            console.print(table)
            console.print(f"[green]📊 Full report saved to: {report_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error generating summary report: {e}[/red]")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SD Export and LoRA Enhancement Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--face", "-f", help="Original face reference image")
    parser.add_argument("--config", "-c", default="sd_export_config.yaml", help="Config file path")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch process directory")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SDExportPipeline(args.config)
    
    try:
        if args.batch or os.path.isdir(args.input):
            # Batch processing
            results = pipeline.batch_process(args.input, args.face)
            console.print(f"[green]✅ Processed {len(results)} images[/green]")
        else:
            # Single image processing
            result = pipeline.process_image(args.input, args.face)
            if result.success:
                console.print(f"[green]✅ Successfully processed: {args.input}[/green]")
                if result.enhanced_image_path:
                    console.print(f"[cyan]🎨 Enhanced image: {result.enhanced_image_path}[/cyan]")
            else:
                console.print(f"[red]❌ Failed to process: {args.input}[/red]")
                console.print(f"[red]Error: {result.error_message}[/red]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Processing interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}[/red]")

if __name__ == "__main__":
    main()
