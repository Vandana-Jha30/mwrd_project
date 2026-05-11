import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(nn.Module):
    """Perceptual Loss using pretrained VGG19 features."""
    def __init__(self, layer='relu3_3', use_normalization=True, device='cpu'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        layers = {'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 16, 'relu4_3': 25}
        assert layer in layers
        self.vgg = vgg[:layers[layer]+1].to(device)
        self.vgg.eval()
        for p in self.vgg.parameters(): p.requires_grad = False
        if use_normalization:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.use_normalization = use_normalization

    def forward(self, pred, target):
        if self.use_normalization:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)
        return F.l1_loss(pred_feat, target_feat)
