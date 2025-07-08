# model.py
# Creates the deep learning model for classifying tumors.

from torch import nn
from torchvision import models

def create_brain_tumour_model(pretrained=True):
    """
    Creates a pre-trained ResNet-18 model and adapts it for our 3-class problem.
    This is an example of "transfer learning".
    
    Args:
        pretrained (bool): If True, loads a model pre-trained on ImageNet.
    """
    
    # 1. Load a powerful, pre-trained ResNet-18 model
    # The 'weights' argument loads the state from its ImageNet training
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # 2. Freeze the pre-trained layers
    # We don't want to change the original layers that are good at detecting
    # general features like edges and textures.
    for param in model.parameters():
        param.requires_grad = False # This "freezes" the layer

    # 3. Replace the final layer (the "fully connected" or "fc" layer)
    # The original ResNet-18 was trained to classify 1000 different things.
    # We need to classify only 3 things (meningioma, glioma, pituitary).
    # So, we replace its final layer with a new one tailored to our task.
    
    # Get the number of input features for the original final layer
    num_features = model.fc.in_features
    
    # Create a new sequence of layers to be our new "head"
    model.fc = nn.Sequential(
        nn.Linear(num_features, 60), # A new layer with 60 units
        nn.ReLU(inplace=True),       # Activation function
        nn.Dropout(0.5),             # Dropout helps prevent overfitting
        nn.Linear(60, 3)             # The final output layer for our 3 classes
    )
    
    return model