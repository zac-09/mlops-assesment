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
        
        # Save image to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name, "JPEG")
            temp_path = temp_file.name
        
        try:
            # Get prediction
            class_id, confidence, class_name = classifier.predict(temp_path)
            
            result = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "status": "success"
            }
            
            logger.info(f"Prediction successful: class_id={class_id}, confidence={confidence:.3f}")
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

# Initialize the model when the module is imported
init()
