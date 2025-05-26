#!/usr/bin/env python3
"""
Cerebrium deployment app for ImageNet classification model.
FastAPI application that serves the ONNX model for inference.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
import time
from io import BytesIO
from PIL import Image
import numpy as np
from model import ONNXImageClassifier
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ImageNet Classification API",
    description="ONNX-based ImageNet classification service",
    version="1.0.0"
)

# Global model instance
classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
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

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ImageNet Classification API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "timestamp": time.time()
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict class for uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction results
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name, "JPEG")
            temp_path = temp_file.name
        
        try:
            # Get prediction
            start_time = time.time()
            class_id, confidence, class_name = classifier.predict(temp_path)
            inference_time = time.time() - start_time
            
            return {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "inference_time": inference_time,
                "filename": file.filename
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/top-k")
async def predict_top_k(file: UploadFile = File(...), k: int = 5):
    """
    Get top-k predictions for uploaded image.
    
    Args:
        file: Uploaded image file
        k: Number of top predictions to return
        
    Returns:
        JSON response with top-k prediction results
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate parameters
        if k < 1 or k > 100:
            raise HTTPException(status_code=400, detail="k must be between 1 and 100")
        
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name, "JPEG")
            temp_path = temp_file.name
        
        try:
            # Get top-k predictions
            start_time = time.time()
            predictions = classifier.get_top_k_predictions(temp_path, k=k)
            inference_time = time.time() - start_time
            
            # Format results
            results = []
            for class_id, confidence, class_name in predictions:
                results.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence
                })
            
            return {
                "predictions": results,
                "inference_time": inference_time,
                "filename": file.filename,
                "k": k
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Top-k prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "ONNX ImageNet Classifier",
        "input_shape": [1, 3, 224, 224],
        "num_classes": 1000,
        "preprocessing": {
            "resize": [224, 224],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225]
        }
    }

@app.get("/stats")
async def get_stats():
    """Get basic service statistics."""
    return {
        "service": "ImageNet Classification",
        "model_loaded": classifier is not None,
        "endpoints": [
            "/predict",
            "/predict/top-k",
            "/health",
            "/model/info"
        ]
    }

if __name__ == "__main__":
    # Run the server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )