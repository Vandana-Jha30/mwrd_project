import torch
import torch.nn as nn
from .wavelet_block import WaveletTransformModule
from .camamba import CAMambaBlock
from .ss2d import SS2D


class MWModule(nn.Module):
    
    def __init__(self, channels):
        super(MWModule, self).__init__()
        self.wavelet = WaveletTransformModule(channels)
        self.camamba = CAMambaBlock(channels)
        self.ss2d = SS2D(d_model=channels)
        # self.fusion = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.res_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Step 1: Wavelet Transform
        wavelet_out = self.wavelet(x)

        # Step 2: CAMamba
        camamba_out = self.camamba(wavelet_out)

        # Parallel to camamba, ss2d integrated
        ss2d_in = wavelet_out.permute(0, 2, 3, 1).contiguous()
        ss2d_out = self.ss2d(ss2d_in).permute(0, 3, 1, 2).contiguous()

        # fuse both outputs
        combined = torch.cat([camamba_out, ss2d_out], dim=1)
        fused = self.fusion(combined)


        # Step 3: Residual addition + Conv2D
        # out = self.fusion(camamba_out + wavelet_out)
        out = self.res_conv(fused + wavelet_out)
        return out + x


class MWRNet(nn.Module):
    
    def __init__(self, in_ch=3, base_ch=64, num_blocks=2, use_refiner=False, refiner=None):
        super(MWRNet, self).__init__()
        self.use_refiner = use_refiner
        self.refiner = refiner

        # Shallow feature extraction
        self.head = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # Stack of MW modules 
        # self.blocks = nn.Sequential(*[MWModule(base_ch) for _ in range(num_blocks)])
        self.mw1 = MWModule(base_ch)
        self.mw2 = MWModule(base_ch)

        # Output reconstruction
        self.tail = nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        res1 = feat
        #skip connection needed
        feat = self.mw1(feat) + res1
        res2 = feat
        #skip connection needed
        feat = self.mw2(feat) + res2

        # Add residual from input
        out = self.tail(feat) 
        out = out + x

        # Optional diffusion refiner (DDPM)
        if self.use_refiner and self.refiner is not None:
            out = self.refiner(out)

        return out
