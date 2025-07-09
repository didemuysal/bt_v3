# model.py
# Creates the deep learning model for classifying tumors.

from torch import nn
from torchvision import models

def create_brain_tumour_model(model_name: str = 'resnet18', pretrained: bool = True):
    """
    Creates a pre-trained model (ResNet18 or ResNet50) and adapts it.

    Args:
        model_name (str): The name of the model architecture ('resnet18' or 'resnet50').
        pretrained (bool): If True, loads a model pre-trained on ImageNet.
    """
    if model_name.lower() == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name.lower() == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose 'resnet18' or 'resnet50'.")

    # Freeze the pre-trained layers for initial head training
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features for the original final layer
    num_features = model.fc.in_features

    # Replace the final layer with a new head based on Rasa et al. (2024)
    # This provides a citable justification for our classifier architecture.
    model.fc = nn.Sequential(
        nn.Linear(num_features, 60),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(60, 3)
    )

    return model