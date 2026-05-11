import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps ** 2))
        return loss
