#!/usr/bin/env python3
"""
PyTorch ImageNet classification model implementation.
This is the exact implementation that matches the provided model weights.
"""

from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from torchvision import transforms
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Classifier(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        layers: List[int] = [2, 2, 2, 2],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def preprocess_numpy(self, img):
        """Original preprocessing method from the provided code."""
        resize = transforms.Resize((224, 224))   # must same as here
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        return img

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
            
            # Use the exact preprocessing from the provided code
            input_tensor = self.preprocess_numpy(image).unsqueeze(0)
            
            # Set model to evaluation mode
            self.eval()
            
            # Perform inference
            with torch.no_grad():
                outputs = self(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get prediction
                confidence, predicted = torch.max(probabilities, 1)
                class_id = predicted.item()
                confidence_score = confidence.item()
                
                return class_id, confidence_score, probabilities.numpy()[0]
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise


# Create the ImageNetClassifier class for backward compatibility
class ImageNetClassifier(Classifier):
    """Alias for Classifier to maintain compatibility with existing code."""
    
    def __init__(self, num_classes=1000):
        """Initialize with ResNet18 architecture (BasicBlock, [2, 2, 2, 2])."""
        super().__init__(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=num_classes
        )


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


# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Test model creation using the exact code from the provided implementation
    logger.info("Creating ImageNet classifier using exact implementation...")
    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    
    # Test with random input
    logger.info("Testing with random input...")
    test_input = torch.randn(1, 3, 224, 224)
    output = mtailor(test_input)
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # If weights file exists, test loading and prediction
    weights_path = "pytorch_model_weights.pth"
    if os.path.exists(weights_path):
        logger.info("Loading pretrained weights...")
        try:
            # Load using the exact method from provided code
            mtailor.load_state_dict(torch.load(weights_path, map_location='cpu'))
            mtailor.eval()
            logger.info("Model loaded successfully!")
            
            # Test with provided images if they exist
            test_images = ["n01440764_tench", "n01667114_mud_turtle", "n01667114_mud_turtle.JPEG"]
            
            for image_path in test_images:
                if os.path.exists(image_path):
                    logger.info(f"Testing prediction on {image_path}...")
                    try:
                        img = Image.open(image_path).convert('RGB')
                        inp = mtailor.preprocess_numpy(img).unsqueeze(0)
                        res = mtailor.forward(inp)
                        predicted_class = torch.argmax(res).item()
                        
                        logger.info(f"Predicted class: {predicted_class}")
                        
                        # Also test with our predict method
                        class_id, confidence, probs = mtailor.predict(image_path)
                        logger.info(f"Using predict method - Class: {class_id}, Confidence: {confidence:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Prediction failed for {image_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
    else:
        logger.warning(f"Weights file not found: {weights_path}")
        logger.info("Download weights from: https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0")

