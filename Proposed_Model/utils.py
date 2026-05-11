import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# ---------------- Checkpoint Utils ----------------
def save_checkpoint(state, path):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved at: {path}")


# ---------------- Image Saving Utils ----------------
def save_image_tensor(tensor, path, normalize=True):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tensor.dim() == 4:
        vutils.save_image(tensor, path, normalize=normalize)
    else:
        vutils.save_image(tensor.unsqueeze(0), path, normalize=normalize)


# ---------------- Tensor → NumPy Conversion ----------------
def tensor_to_ndimage(tensor):
    
    img = tensor.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return img.astype(np.float32)


# ---------------- Image Quality Metrics ----------------
def compute_psnr_ssim(pred, tgt):
    
    psnr = psnr_metric(tgt, pred, data_range=1.0)
    ssim = ssim_metric(tgt, pred, channel_axis=2, data_range=1.0)
    return psnr, ssim


# ---------------- Plotting Training & Validation Curves ----------------
def plot_training_curves(history, out_path):
    
    epochs = range(1, len(history.get('val_loss', [])) + 1)

    plt.figure(figsize=(18, 6))

    # ---- Validation Loss ----
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.get('val_loss', []), 'o-', color='red', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curve')
    plt.legend()
    plt.grid(True)

    # ---- PSNR and SSIM ----
    plt.subplot(1, 3, 2)
    if 'val_psnr' in history:
        plt.plot(epochs, history['val_psnr'], 'o-', color='blue', label='PSNR')
    if 'val_ssim' in history:
        plt.plot(epochs, history['val_ssim'], 'o-', color='green', label='SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.title('PSNR / SSIM Trend')
    plt.legend()
    plt.grid(True)

    # ---- UCIQE and UIQM ----
    plt.subplot(1, 3, 3)
    if 'val_uciqe' in history:
        plt.plot(epochs, history['val_uciqe'], 'o-', color='purple', label='UCIQE')
    if 'val_uiqm' in history:
        plt.plot(epochs, history['val_uiqm'], 'o-', color='orange', label='UIQM')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Underwater Image Quality Metrics')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved full training curve plot to: {out_path}")



def plot_live_metrics(history):
    
    from IPython.display import clear_output, display
    clear_output(wait=True)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.get('val_loss', []), 'r-', label='Val Loss')
    plt.legend(); plt.grid(True); plt.title("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(history.get('val_psnr', []), 'b-', label='PSNR')
    plt.plot(history.get('val_ssim', []), 'g-', label='SSIM')
    plt.legend(); plt.grid(True); plt.title("PSNR / SSIM")

    plt.subplot(1, 3, 3)
    plt.plot(history.get('val_uciqe', []), 'm-', label='UCIQE')
    plt.plot(history.get('val_uiqm', []), 'y-', label='UIQM')
    plt.legend(); plt.grid(True); plt.title("UCIQE / UIQM")

    plt.tight_layout()
    display(plt.gcf())
    plt.close()