import torch.nn as nn
import torch

img = torch.randn((1, 3, 224, 224))

img = nn.Conv2d(3, 3, 3, 1)(img)

print(img.shape)
