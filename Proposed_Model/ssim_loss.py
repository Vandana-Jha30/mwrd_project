import torch.nn.functional as F

def ssim_loss(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """Channel-wise SSIM loss, returns 1-SSIM. Inputs: BxCxHxW in [0,1]."""
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    sigma_x = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_x * mu_y
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = SSIM_n / (SSIM_d + 1e-8)
    return 1 - ssim_map.mean()
