#!/usr/bin/env python3
"""
Main entry point for Cerebrium deployment.
This file defines the prediction function that Cerebrium will call.
"""

import os
import tempfile
import logging
from model import ONNXImageClassifier
from PIL import Image
import base64
import io
import numpy as np
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
classifier = None

def init():
    """Initialize the model when the container starts."""
    global classifier
    try:
        model_path = "model.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info("Loading ONNX model...")
        classifier = ONNXImageClassifier(model_path)
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def preprocess_image_exact(image):
    """
    Apply the exact same preprocessing as the original PyTorch implementation.
    This ensures perfect consistency between PyTorch and ONNX inference.
    """
    try:
        # Use the exact same preprocessing pipeline as pytorch_model.py
        resize = transforms.Resize((224, 224))
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # Apply transforms in exact order
        image = resize(image)
        image = crop(image)
        image = to_tensor(image)
        image = normalize(image)
        
        # Add batch dimension and convert to numpy for ONNX
        image_array = image.unsqueeze(0).numpy()
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error in exact preprocessing: {str(e)}")
        raise

def predict(item):
    """
    Main prediction function called by Cerebrium.
    
    Args:
        item: Input data containing image
        
    Returns:
        dict: Prediction results
    """
    global classifier
    
    if classifier is None:
        return {"error": "Model not loaded"}
    
    try:
        # Handle different input formats
        if isinstance(item, dict):
            if "image" in item:
                image_data = item["image"]
            elif "file" in item:
                image_data = item["file"]
            else:
                return {"error": "No image data found in request"}
        else:
            image_data = item
        
        # Handle base64 encoded images
        if isinstance(image_data, str):
            try:
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                return {"error": f"Failed to decode base64 image: {str(e)}"}
        else:
            # Handle file uploads or raw bytes
            try:
                if hasattr(image_data, 'read'):
                    image_bytes = image_data.read()
                else:
                    image_bytes = image_data
                
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                return {"error": f"Failed to open image: {str(e)}"}
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Method 1: Use the exact preprocessing pipeline (recommended)
        try:
            # Apply exact same preprocessing as PyTorch model
            input_tensor = preprocess_image_exact(image)
            
            # Run inference directly with ONNX session
            outputs = classifier.session.run([classifier.output_name], {classifier.input_name: input_tensor})
            predictions = outputs[0][0]  # Remove batch dimension
            
            # Apply softmax to get proper probabilities
            exp_predictions = np.exp(predictions - np.max(predictions))  # For numerical stability
            probabilities = exp_predictions / np.sum(exp_predictions)
            
            # Get prediction
            class_id = int(np.argmax(probabilities))
            confidence = float(probabilities[class_id])
            class_name = f"class_{class_id}"
            
        except Exception as preprocessing_error:
            logger.warning(f"Exact preprocessing failed: {preprocessing_error}, falling back to file-based method")
            
            # Method 2: Fallback to file-based prediction (for compatibility)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file.name, "JPEG")
                temp_path = temp_file.name
            
            try:
                # Use the model's predict method
                class_id, confidence, class_name = classifier.predict(temp_path)
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
        
        result = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "status": "success"
        }
        
        logger.info(f"Prediction successful: class_id={class_id}, confidence={confidence:.3f}")
        return result
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

# Initialize the model when the module is imported
init()