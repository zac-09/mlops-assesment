#!/usr/bin/env python3
"""
ONNX Model wrapper and image preprocessing classes for ImageNet classification.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import logging
from typing import Tuple, List, Union
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Handles image preprocessing for ImageNet classification model.
    Implements the specific preprocessing steps required by the model.
    """
    
    def __init__(self):
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.target_size = (224, 224)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            np.ndarray: Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 224x224 using bilinear interpolation
            image = image.resize(self.target_size, Image.BILINEAR)
            
            # Convert to numpy array and normalize to [0, 1]
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Normalize using ImageNet mean and std
            image_array = (image_array - self.mean) / self.std
            
            # Convert from HWC to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def preprocess_numpy(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess numpy array image for model inference.
        
        Args:
            image_array (np.ndarray): Input image as numpy array (H, W, C)
            
        Returns:
            np.ndarray: Preprocessed image tensor
        """
        try:
            # Ensure float32 and normalize to [0, 1] if needed
            if image_array.dtype != np.float32:
                image_array = image_array.astype(np.float32)
            
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
            
            # Resize if needed
            if image_array.shape[:2] != self.target_size:
                from PIL import Image
                image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
                image_pil = image_pil.resize(self.target_size, Image.BILINEAR)
                image_array = np.array(image_pil, dtype=np.float32) / 255.0
            
            # Normalize using ImageNet mean and std
            image_array = (image_array - self.mean) / self.std
            
            # Convert from HWC to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing numpy image: {str(e)}")
            raise

class ONNXImageClassifier:
    """
    ONNX model wrapper for ImageNet classification.
    Handles model loading, inference, and result processing.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize ONNX model.
        
        Args:
            model_path (str): Path to ONNX model file
        """
        self.model_path = model_path
        self.preprocessor = ImagePreprocessor()
        self.session = None
        self.input_name = None
        self.output_name = None
        
        # Load ImageNet class names
        self.class_names = self._load_imagenet_classes()
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and initialize session."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Set up ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Model loaded successfully: {self.model_path}")
            logger.info(f"Input name: {self.input_name}")
            logger.info(f"Output name: {self.output_name}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_imagenet_classes(self) -> List[str]:
        """Load ImageNet class names."""
        # Simplified class names - in production, load from file
        return [f"class_{i}" for i in range(1000)]
    
    def predict(self, image_path: str) -> Tuple[int, float, str]:
        """
        Predict class for input image.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            Tuple[int, float, str]: (class_id, confidence, class_name)
        """
        try:
            # Preprocess image
            input_tensor = self.preprocessor.preprocess_image(image_path)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            predictions = outputs[0][0]  # Remove batch dimension
            
            # Get top prediction
            class_id = int(np.argmax(predictions))
            confidence = float(np.max(predictions))
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"
            
            return class_id, confidence, class_name
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[int, float, str]]:
        """
        Predict classes for multiple images.
        
        Args:
            image_paths (List[str]): List of image paths
            
        Returns:
            List[Tuple[int, float, str]]: List of (class_id, confidence, class_name)
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {str(e)}")
                results.append((-1, 0.0, "error"))
        
        return results
    
    def get_top_k_predictions(self, image_path: str, k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Get top-k predictions for input image.
        
        Args:
            image_path (str): Path to input image
            k (int): Number of top predictions to return
            
        Returns:
            List[Tuple[int, float, str]]: List of (class_id, confidence, class_name)
        """
        try:
            # Preprocess image
            input_tensor = self.preprocessor.preprocess_image(image_path)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            predictions = outputs[0][0]  # Remove batch dimension
            
            # Get top-k predictions
            top_k_indices = np.argsort(predictions)[-k:][::-1]
            
            results = []
            for idx in top_k_indices:
                class_id = int(idx)
                confidence = float(predictions[idx])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"
                results.append((class_id, confidence, class_name))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during top-k prediction: {str(e)}")
            raise

# Factory function for easy model creation
def create_classifier(model_path: str = "model.onnx") -> ONNXImageClassifier:
    """
    Factory function to create ONNX classifier instance.
    
    Args:
        model_path (str): Path to ONNX model file
        
    Returns:
        ONNXImageClassifier: Initialized classifier instance
    """
    return ONNXImageClassifier(model_path)