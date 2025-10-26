import torch
import torch.nn as nn
from torchvision import models


def get_model(backbone: str, num_classes: int = 10, pretrained: bool = False):
    
    if(backbone == "resnet18"):
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    elif(backbone == "resnet34"):
        model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
    elif(backbone == "resnet50"):
        model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model 