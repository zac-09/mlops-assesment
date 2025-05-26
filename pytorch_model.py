#!/usr/bin/env python3
"""
PyTorch ImageNet classification model implementation.
This matches the actual ResNet architecture of the provided weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageNetClassifier(nn.Module):
    """
    PyTorch ImageNet classification model using ResNet18 architecture.
    This matches the structure of the provided model weights.
    """
    
    def __init__(self, num_classes=1000):
        """
        Initialize the ResNet-based model.
        
        Args:
            num_classes (int): Number of output classes (default: 1000 for ImageNet)
        """
        super(ImageNetClassifier, self).__init__()
        
        # Use ResNet18 architecture to match the provided weights
        self.resnet = models.resnet18(pretrained=False)
        
        # Modify the final layer to ensure correct number of classes
        if self.resnet.fc.out_features != num_classes:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.resnet(x)

    def preprocess_numpy(self, image_array):
        """
        Preprocess numpy image array for inference.
        
        Args:
            image_array (np.ndarray): Input image array (H, W, C) in range [0, 255]
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image_array, np.ndarray):
                if image_array.dtype != np.uint8:
                    image_array = (image_array * 255).astype(np.uint8)
                image = Image.fromarray(image_array)
            else:
                image = image_array
            
            # Define preprocessing transforms
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Apply preprocessing
            tensor = preprocess(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def predict(self, image_path):
        """
        Predict class for image file.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (class_id, confidence, probabilities)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            input_tensor = preprocess(image).unsqueeze(0)
            
            # Set model to evaluation mode
            self.eval()
            
            # Perform inference
            with torch.no_grad():
                outputs = self(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get prediction
                confidence, predicted = torch.max(probabilities, 1)
                class_id = predicted.item()
                confidence_score = confidence.item()
                
                return class_id, confidence_score, probabilities.numpy()[0]
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

def load_model(weights_path="pytorch_model_weights.pth"):
    """
    Load pretrained model from weights file.
    
    Args:
        weights_path (str): Path to model weights
        
    Returns:
        ImageNetClassifier: Loaded model
    """
    try:
        model = ImageNetClassifier()
        
        # Load weights
        if torch.cuda.is_available():
            state_dict = torch.load(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location='cpu')
        
        # The weights might be wrapped in a 'state_dict' key or be direct
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        logger.info(f"Model loaded successfully from {weights_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def inspect_model_weights(weights_path="pytorch_model_weights.pth"):
    """
    Inspect the structure of the model weights to understand the architecture.
    
    Args:
        weights_path (str): Path to model weights
    """
    try:
        if torch.cuda.is_available():
            state_dict = torch.load(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location='cpu')
        
        # Handle different weight file formats
        if 'state_dict' in state_dict:
            weights = state_dict['state_dict']
        elif 'model' in state_dict:
            weights = state_dict['model']
        else:
            weights = state_dict
        
        logger.info("Model weight keys:")
        for key in sorted(weights.keys()):
            logger.info(f"  {key}: {weights[key].shape}")
            
        # Determine if it's ResNet architecture
        if any(key.startswith('layer') for key in weights.keys()):
            logger.info("Detected ResNet architecture")
        elif any(key.startswith('features') for key in weights.keys()):
            logger.info("Detected VGG-like architecture")
        else:
            logger.info("Unknown architecture")
            
    except Exception as e:
        logger.error(f"Error inspecting weights: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Test model creation
    logger.info("Creating ImageNet classifier...")
    model = ImageNetClassifier()
    
    # Test with random input
    logger.info("Testing with random input...")
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test preprocessing
    logger.info("Testing preprocessing...")
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    preprocessed = model.preprocess_numpy(test_image)
    logger.info(f"Preprocessed shape: {preprocessed.shape}")
    
    # If weights file exists, inspect and test loading
    weights_path = "pytorch_model_weights.pth"
    if os.path.exists(weights_path):
        logger.info("Inspecting model weights...")
        inspect_model_weights(weights_path)
        
        logger.info("Loading pretrained weights...")
        try:
            model = load_model(weights_path)
            logger.info("Model loaded successfully!")
            
            # Test with provided images if they exist
            test_images = ["n01440764_tench", "n01667114_mud_turtle"]
            
            for image_path in test_images:
                if os.path.exists(image_path):
                    logger.info(f"Testing prediction on {image_path}...")
                    try:
                        class_id, confidence, probs = model.predict(image_path)
                        logger.info(f"Predicted class: {class_id}, Confidence: {confidence:.4f}")
                    except Exception as e:
                        logger.error(f"Prediction failed for {image_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
    else:
        logger.warning(f"Weights file not found: {weights_path}")
        logger.info("Download weights from: https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0")