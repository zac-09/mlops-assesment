#!/usr/bin/env python3
"""
Convert PyTorch model to ONNX format for efficient deployment.
"""

import torch
import torch.onnx
import onnx
from pytorch_model import Classifier, BasicBlock
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_weights():
    """Download model weights if not present."""
    weights_path = "pytorch_model_weights.pth"
    if not os.path.exists(weights_path):
        logger.info("Downloading model weights...")
        import urllib.request
        url = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"
        try:
            urllib.request.urlretrieve(url, weights_path)
            # Verify file size
            if os.path.getsize(weights_path) > 1000000:  # Should be > 1MB
                logger.info("Weights downloaded successfully!")
            else:
                logger.error("Downloaded file seems too small")
                os.remove(weights_path)
                raise Exception("Invalid download")
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            logger.info("Please manually download from: https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0")
            raise
    return weights_path

def convert_to_onnx(pytorch_model_path, onnx_model_path):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        pytorch_model_path (str): Path to PyTorch model weights
        onnx_model_path (str): Output path for ONNX model
    """
    try:
        # Load PyTorch model using the exact implementation
        logger.info("Creating PyTorch model...")
        model = Classifier(BasicBlock, [2, 2, 2, 2])  # ResNet18 architecture
        
        # Load weights with proper error handling
        logger.info("Loading model weights...")
        if torch.cuda.is_available():
            state_dict = torch.load(pytorch_model_path)
        else:
            state_dict = torch.load(pytorch_model_path, map_location='cpu')
        
        # Handle different weight file formats
        if 'state_dict' in state_dict:
            weights = state_dict['state_dict']
        elif 'model' in state_dict:
            weights = state_dict['model']
        else:
            weights = state_dict
        
        # Load the state dict - should work perfectly now with exact architecture
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=True)
        
        if missing_keys:
            logger.warning(f"Missing keys in state_dict: {missing_keys}")
            raise Exception("Model architecture mismatch - missing keys")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
            raise Exception("Model architecture mismatch - unexpected keys")
        
        model.eval()
        logger.info("Model loaded successfully with exact architecture match!")
        
        # Create dummy input tensor (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Test the model with dummy input first
        logger.info("Testing model with dummy input...")
        with torch.no_grad():
            test_output = model(dummy_input)
            logger.info(f"Model output shape: {test_output.shape}")
            logger.info(f"Output range: [{test_output.min().item():.3f}, {test_output.max().item():.3f}]")
        
        # Export to ONNX
        logger.info("Converting to ONNX format...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        # Verify the ONNX model
        logger.info("Verifying ONNX model...")
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"Successfully converted model to ONNX: {onnx_model_path}")
        
        # Print model info
        logger.info(f"Input shape: {dummy_input.shape}")
        logger.info(f"Model file size: {os.path.getsize(onnx_model_path) / (1024*1024):.2f} MB")
        
        # Test ONNX model to ensure it works
        logger.info("Testing ONNX model...")
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run test inference
        onnx_output = session.run([output_name], {input_name: dummy_input.numpy()})
        logger.info(f"ONNX output shape: {onnx_output[0].shape}")
        
        # Compare PyTorch vs ONNX outputs
        pytorch_output = test_output.numpy()
        onnx_result = onnx_output[0]
        max_diff = abs(pytorch_output - onnx_result).max()
        logger.info(f"Max difference between PyTorch and ONNX: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            logger.info("✅ PyTorch and ONNX outputs match perfectly!")
        else:
            logger.warning(f"⚠️ Small difference detected: {max_diff:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        return False

def main():
    """Main conversion function."""
    # Download weights if needed
    pytorch_weights_path = download_weights()
    
    # Define output path
    onnx_model_path = "model.onnx"
    
    # Convert model
    success = convert_to_onnx(pytorch_weights_path, onnx_model_path)
    
    if success:
        logger.info("🎉 Conversion completed successfully!")
        logger.info("The ONNX model is ready for deployment!")
    else:
        logger.error("❌ Conversion failed!")
        exit(1)

if __name__ == "__main__":
    main()