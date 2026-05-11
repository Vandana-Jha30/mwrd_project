import torch
import torch.nn as nn
import torch.nn.functional as F
from .ss2d import SS2D  


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CurveCALayer(nn.Module):
    def __init__(self, channel):
        super(CurveCALayer, self).__init__()
        self.n_curve = 3
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, stride=1, padding=0), nn.Sigmoid()
        )

    def forward(self, x):
        a = self.predict_a(x)
        x = self.relu(x) - self.relu(x - 1)
        for i in range(self.n_curve):
            x = x + a[:, i:i + 1] * x * (1 - x)
        return x


class CAMambaBlock(nn.Module):
    

    def __init__(self, channels, expansion=2, s2d_state=16, s2d_expand=2, dropout=0.0):
        super().__init__()
        hidden = channels * expansion

        # ---- Main branch ----
        self.ln1 = nn.LayerNorm(channels)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.SiLU()
        self.ss2d = SS2D(d_model=hidden, d_state=s2d_state, expand=s2d_expand, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1)

        # ---- Parallel Linear-SiLU path ----
        self.linear_skip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU()
        )

        self.merge_linear = nn.Conv2d(channels, channels, kernel_size=1)

        # ---- Post-processing ----
        self.ln3 = nn.LayerNorm(channels)
        self.ca = CALayer(channels, reduction=16)
        self.res_scale = 1.0

    def forward(self, x):
        B, C, H, W = x.shape

        # ---- Main Branch ----
        y = x.permute(0, 2, 3, 1)               # B, H, W, C
        y = self.ln1(y)
        y = y.permute(0, 3, 1, 2).contiguous()  # B, C, H, W

        y = self.pw1(y)
        y = self.dwconv(y)
        y = self.act(y)

        # SS2D operates in (B, H, W, C)
        y = y.permute(0, 2, 3, 1).contiguous()
        y = self.ss2d(y)
        y = y.permute(0, 3, 1, 2).contiguous()

        y = self.ln2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        y = self.pw2(y)

        # ---- Parallel Linear Path ----
        skip = self.linear_skip(x)

        # ---- Merge ----
        out = y + skip
        
        #add linear layer !!
        out = self.merge_linear(out)

        # ---- LayerNorm + Channel Attention ----
        out = self.ln3(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        out = self.ca(out)

        # ---- Residual ----
        return out
