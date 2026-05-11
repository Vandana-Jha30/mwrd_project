import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import PairedImageDataset
from models.mwrnet import MWRNet
from utils import save_checkpoint, tensor_to_ndimage, compute_psnr_ssim
from losses import CharbonnierLoss
from metrics.uciqe import torch_uciqe
from metrics.uiqm import torch_uiqm
from models.ddpm import Unet, GaussianDiffusion
from PIL import Image, ImageDraw, ImageFont

# ---------------- Paths ----------------
if '__file__' in globals():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    ROOT_DIR = os.getcwd()


default_train_input = os.path.join(ROOT_DIR, 'Dataset/train/input')
default_train_target = os.path.join(ROOT_DIR, 'Dataset/train/target')
default_val_input = os.path.join(ROOT_DIR, 'Dataset/val/input')
default_val_target = os.path.join(ROOT_DIR, 'Dataset/val/target')
default_ckpt_dir = os.path.join(ROOT_DIR, 'checkpoints')
default_out_dir = os.path.join(ROOT_DIR, 'results')

# ---------------- GPU Setup ----------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def save_labeled_grid(images, labels, save_path, spacing=25, font_size=24):
    """Save horizontal grid of input, output, GT with labels."""
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + spacing * (len(images) - 1)
    max_height = max(heights) + font_size + 10

    new_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(new_img)
    x_offset = 0

    for img, label in zip(images, labels):
        new_img.paste(img, (x_offset, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x_offset + (img.width - text_w) // 2
        text_y = img.height + 5
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
        x_offset += img.width + spacing

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    new_img.save(save_path)



def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---------------- Dataset ----------------
    train_ds = PairedImageDataset(args.train_input, args.train_target, resize=(args.size, args.size))
    val_ds = PairedImageDataset(args.val_input, args.val_target, resize=(args.size, args.size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # ---------------- Model ----------------
    print("\nInitializing Diffusion Refiner and MWRNet...")
    refiner_unet = Unet(dim=64, channels=3, dim_mults=(1, 2, 4, 8)).to(device)
    refiner = GaussianDiffusion(
        refiner_unet, image_size=args.size, timesteps=1000,
        sampling_timesteps=250, loss_type='l1', beta_schedule='sigmoid'
    ).to(device)

    model = MWRNet(in_ch=3, base_ch=args.base_ch, num_blocks=args.num_blocks,
                   use_refiner=True, refiner=refiner).to(device)

    # ---------------- Loss & Optimizer ----------------
    l1 = nn.L1Loss()
    charbonnier = CharbonnierLoss()
    def criterion(pred, target):
        return 0.8 * l1(pred, target) + 0.2 * charbonnier(pred, target)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scaler = torch.cuda.amp.GradScaler()

    # ---------------- Auto Resume ----------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.ckpt_dir, "best_model_psnr.pth")

    if args.resume:
        resume_path = args.resume
    elif os.path.exists(best_ckpt_path):
        resume_path = best_ckpt_path
        print(f"Found best_model_psnr.pth → Auto-resuming from {resume_path}")
    else:
        resume_path = None

    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_uciqe': [], 'val_uiqm': []}
    best_val_loss, best_psnr, best_ssim = float('inf'), 0, 0
    start_epoch = 0

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        history = ckpt.get('history', history)
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting new training")

    # ---------------- Training ----------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inp, tgt in pbar:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(inp)
                if out.min() < 0:
                    out = (out + 1) / 2
                out = out.clamp(0, 1)
                loss = criterion(out, tgt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())
            pbar.set_postfix({"train_loss": np.mean(epoch_losses)})

        avg_train_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_train_loss)

        # ---------------- Validation ----------------
        model.eval()
        val_losses, psnrs, ssims, uciqes, uiqms = [], [], [], [], []
        os.makedirs(args.out_dir, exist_ok=True)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            for i, (inp, tgt) in enumerate(val_loader):
                inp, tgt = inp.to(device), tgt.to(device)
                out = model(inp)
                out = (out + 1) / 2 if out.min() < 0 else out
                out = out.clamp(0, 1)

                val_loss = criterion(out, tgt).item()
                val_losses.append(val_loss)

                out_np = tensor_to_ndimage(out[0])
                tgt_np = tensor_to_ndimage(tgt[0])
                psnr, ssim = compute_psnr_ssim(out_np, tgt_np)

                out_for_metrics = out[0].detach().cpu().float()
                uciqe = torch_uciqe(out_for_metrics).item()
                uiqm = torch_uiqm(out_for_metrics).item()

                psnrs.append(psnr)
                ssims.append(ssim)
                uciqes.append(uciqe)
                uiqms.append(uiqm)

                # Save comparison image
                if (epoch + 1) % 15 == 0:
      
                    inp_img = Image.fromarray((inp[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    out_img = Image.fromarray((out[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    tgt_img = Image.fromarray((tgt[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))

                    
                    save_path = os.path.join(args.out_dir, f"comparison_{i+1:04d}.png")

                    save_labeled_grid(
                        [inp_img, out_img, tgt_img],
                        ["Input Image", "Output Image", "Ground Truth"],
                        save_path
                    )
                # inp_img = Image.fromarray((inp[0].detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8))
                # out_img = Image.fromarray((out[0].detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8))
                # tgt_img = Image.fromarray((tgt[0].detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8))
                # save_labeled_grid([inp_img, out_img, tgt_img],
                #                   ["Input Image", "Output Image", "Ground Truth"],
                #                   os.path.join(args.out_dir, f"comparison_{i+1:04d}_epoch{epoch+1}.png"))

        avg_val_loss = np.mean(val_losses)
        avg_psnr, avg_ssim = np.mean(psnrs), np.mean(ssims)
        avg_uciqe, avg_uiqm = np.mean(uciqes), np.mean(uiqms)

        history['val_loss'].append(avg_val_loss)
        history['val_psnr'].append(avg_psnr)
        history['val_ssim'].append(avg_ssim)
        history['val_uciqe'].append(avg_uciqe)
        history['val_uiqm'].append(avg_uiqm)

        scheduler.step(avg_val_loss)

        print(f"\nValidation → Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}, "
              f"UCIQE: {avg_uciqe:.3f}, UIQM: {avg_uiqm:.3f}")

        # ---------------- Checkpoints ----------------
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history
        }
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(args.ckpt_dir, "best_model_loss.pth"))
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(checkpoint, os.path.join(args.ckpt_dir, "best_model_psnr.pth"))
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(checkpoint, os.path.join(args.ckpt_dir, "best_model_ssim.pth"))

        # ---------------- Plot Metrics Every 10 Epochs ----------------
        if (epoch + 1) % 15 == 0:
            epochs_range = range(1, len(history['val_loss']) + 1)
            plt.figure(figsize=(20, 8))

            plt.subplot(1, 5, 1)
            plt.plot(epochs_range, history['train_loss'], 'o-', color='blue', label='Train Loss')
            plt.plot(epochs_range, history['val_loss'], 'o-', color='red', label='Val Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train vs Validation Loss'); plt.legend(); plt.grid(True)

            plt.subplot(1, 5, 2)
            plt.plot(epochs_range, history['val_psnr'], 'o-', color='green', label='Val PSNR')
            plt.xlabel('Epoch'); plt.ylabel('PSNR (dB)')
            plt.title('Validation PSNR')
            plt.legend(); plt.grid(True)

            plt.subplot(1, 5, 3)
            plt.plot(epochs_range, history['val_ssim'], 'o-', color='red', label='Val SSIM')
            plt.xlabel('Epoch'); plt.ylabel('SSIM')
            plt.title('Validation SSIM')
            plt.legend(); plt.grid(True)

            plt.subplot(1, 5, 4)
            plt.plot(epochs_range, history['val_uciqe'], 'o-', color='purple', label='UCIQE')
            plt.xlabel('Epoch'); plt.ylabel('Score')
            plt.title('UCIQE')
            plt.legend(); plt.grid(True)

            plt.subplot(1, 5, 5)
            plt.plot(epochs_range, history['val_uiqm'], 'o-', color='orange', label='UIQM')
            plt.xlabel('Epoch'); plt.ylabel('Score')
            plt.title('UIQM')
            plt.legend(); plt.grid(True)

            plt.tight_layout()
            save_path = os.path.join(args.ckpt_dir, f"metrics_epoch_{epoch+1}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved training graph → {save_path}")

    print("\nTraining Completed Successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input', default=default_train_input)
    parser.add_argument('--train_target', default=default_train_target)
    parser.add_argument('--val_input', default=default_val_input)
    parser.add_argument('--val_target', default=default_val_target)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--ckpt_dir', default=default_ckpt_dir)
    parser.add_argument('--out_dir', default=default_out_dir)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    train(args)

