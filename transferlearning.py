from torchvision.models import resnet50, ResNet50_Weights
import torch
model = resnet50(weights = ResNet50_Weights.DEFAULT)

image = torch.rand(2,3,224,224)
output = model(image)
print(output.shape)
