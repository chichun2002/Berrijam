import torchvision
import torch

model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')

torch.save(model, 'Pretrained_Models/vit_B')