import torch
import torchvision.models as models
from torchsummary import summary

# Load the pretrained ResNet-50 model
resnet50 = models.resnet50(pretrained=True)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

# Use the torchsummary package to visualize the network
summary(resnet50, input_size=(3, 224, 224))
