import torchvision
import torch

<<<<<<< Updated upstream
model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')

torch.save(model, 'Pretrained_Models/vit_B')
=======
model_mnasnet = torchvision.models.mnasnet0_75(weights='IMAGENET1K_V1')
model_efficientnet = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
model_shufflenet = torchvision.models.shufflenet_v2_x1_5(weights='IMAGENET1K_V1')
model_regnet = torchvision.models.regnet_y_400mf(weights='IMAGENET1K_V1')
model_vit = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')

torch.save(model_mnasnet, 'Pretrained_Models/mnasnet')
torch.save(model_efficientnet, 'Pretrained_Models/efficientnet')
torch.save(model_shufflenet, 'Pretrained_Models/shufflenet')
torch.save(model_regnet, 'Pretrained_Models/regnet')

# model_vit = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
torch.save(model_vit, 'Pretrained_Models/vit_b')
>>>>>>> Stashed changes
