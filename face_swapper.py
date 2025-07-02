import onnxruntime
from PIL import Image
import numpy as np
import insightface
import cv2
from typing import List, Tuple

class InsightFaceSwapper:
    def __init__(self, model_path: str):
        """
        Initialize the InsightFaceSwapper with a face swapping ONNX model.
        
        Args:
            model_path: Path to the ONNX face swapping model
        """
        self.model_path = model_path
        self.session = onnxruntime.InferenceSession(model_path)
        
        # Initialize face analysis for detection and alignment
        self.face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l')
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for the ONNX model.
        
        Args:
            face_img: Face image as numpy array
            
        Returns:
            Preprocessed face image ready for model inference
        """
        # Resize to model input size (typically 128x128 or 256x256)
        target_size = (self.input_shape[2], self.input_shape[3]) if len(self.input_shape) == 4 else (128, 128)
        face_img = cv2.resize(face_img, target_size)
        
        # Normalize to [0, 1] and convert to float32
        face_img = face_img.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format
        face_img = np.transpose(face_img, (2, 0, 1))
        
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
        
    def _extract_face_region(self, image: np.ndarray, face) -> np.ndarray:
        """
        Extract and align face region from image.
        
        Args:
            image: Source image
            face: Face detection result from InsightFace
            
        Returns:
            Aligned face image
        """
        # Use InsightFace's face alignment
        aligned_face = insightface.utils.face_align.norm_crop(image, face.kps, image_size=112)
        return aligned_face
        
    def _blend_faces(self, source_face: np.ndarray, target_img: np.ndarray, 
                    target_face, blend_ratio: float = 0.8) -> np.ndarray:
        """
        Blend the swapped face with the target image.
        
        Args:
            source_face: Generated face from the model
            target_img: Target image
            target_face: Target face detection result
            blend_ratio: Blending ratio for seamless integration
            
        Returns:
            Image with blended face
        """
        # Get target face bounding box
        bbox = target_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Resize source face to match target face size
        target_face_size = (x2 - x1, y2 - y1)
        source_face_resized = cv2.resize(source_face, target_face_size)
        
        # Create a copy of target image
        result_img = target_img.copy()
        
        # Simple replacement (could be enhanced with more sophisticated blending)
        result_img[y1:y2, x1:x2] = source_face_resized
        
        return result_img

    def swap_faces(self, source_face_path: str, target_image: Image.Image) -> Image.Image:
        """
        Swap faces between source and target images using the ONNX model.
        
        Args:
            source_face_path: Path to source face image
            target_image: Target PIL Image
            
        Returns:
            PIL Image with swapped faces
        """
        # Load and convert source image
        source_image = Image.open(source_face_path)
        if source_image.mode != 'RGB':
            source_image = source_image.convert('RGB')
        source_img_np = np.array(source_image)
        
        # Convert target image to numpy array
        if target_image.mode != 'RGB':
            target_image = target_image.convert('RGB')
        target_img_np = np.array(target_image)
        
        # Detect faces in both images
        source_faces = self.face_analyzer.get(source_img_np)
        target_faces = self.face_analyzer.get(target_img_np)
        
        if not source_faces:
            raise ValueError("No faces detected in source image")
        if not target_faces:
            raise ValueError("No faces detected in target image")
            
        # Use the first detected face from each image
        source_face = source_faces[0]
        target_face = target_faces[0]
        
        # Extract and align source face
        source_face_aligned = self._extract_face_region(source_img_np, source_face)
        
        # Preprocess for model inference
        source_face_preprocessed = self._preprocess_face(source_face_aligned)
        
        # Run inference with ONNX model
        try:
            model_output = self.session.run(
                [self.output_name], 
                {self.input_name: source_face_preprocessed}
            )[0]
            
            # Post-process model output
            generated_face = model_output[0]  # Remove batch dimension
            
            # Convert from CHW to HWC format
            if generated_face.shape[0] == 3:  # CHW format
                generated_face = np.transpose(generated_face, (1, 2, 0))
                
            # Denormalize from [0,1] to [0,255]
            generated_face = (generated_face * 255).astype(np.uint8)
            
        except Exception as e:
            # Fallback to direct face replacement if model inference fails
            print(f"Model inference failed: {e}. Using direct face replacement.")
            generated_face = source_face_aligned
        
        # Blend the generated face with target image
        result_img = self._blend_faces(generated_face, target_img_np, target_face)
        
        # Convert back to PIL Image
        return Image.fromarray(result_img)

