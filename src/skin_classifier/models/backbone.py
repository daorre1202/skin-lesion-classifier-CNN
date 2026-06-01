"""
Pretrained CNN backbones adapted for 7-class skin lesion classification.

All three architectures are initialised from ImageNet-1K weights and
fine-tuned end-to-end. Only the final classification head is replaced
to output 7 classes.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,       ResNet50_Weights,
    densenet121,    DenseNet121_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
)

SUPPORTED_MODELS = ['resnet50', 'densenet121', 'efficientnet_b3']


def build_model(name: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Instantiate a pretrained backbone with a replaced classification head.

    Args:
        name:        one of 'resnet50', 'densenet121', 'efficientnet_b3'
        num_classes: number of output classes
        device:      target device

    Returns:
        Model moved to device with the new classification head.
    """
    if name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == 'densenet121':
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif name == 'efficientnet_b3':
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    else:
        raise ValueError(
            f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}"
        )

    return model.to(device)
