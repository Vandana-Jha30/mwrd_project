import torch
import torch.nn as nn


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch = in_batch
    out_channel = int(in_channel / (r ** 2))
    out_height, out_width = r * in_height, r * in_width

    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def forward(self, x):
        return iwt_init(x)

#    Channel Attention   #
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


#     Spatial Attention       #

class SALayer(nn.Module):
    
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel pooling (max + avg)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]
        y = self.conv_du(channel_pool)
        return x * y


#     Curved Channel Attn     #

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



#   Wavelet Transform Block   #

class WaveletTransformModule(nn.Module):
   
    def __init__(self, in_channels):
        super(WaveletTransformModule, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        self.conv1 = nn.Conv2d(in_channels * 4, in_channels, 3, padding=1)
        self.curved_attention = CurveCALayer(in_channels)
        self.spatial_attention = SALayer()
        self.channel_attention = CALayer(in_channels, reduction=16)
        self.conv2 = nn.Conv2d(in_channels, in_channels * 4, 3, padding=1)

    def forward(self, x):
        residual = x
        y = self.dwt(x)
        dwt_res = y
        y = self.conv1(y)

        # Applying both attentions (additive fusion) + added channel attention layer
        y = self.curved_attention(y) + self.spatial_attention(y) + self.channel_attention(y)
        y = self.conv2(y)

        #skip connection needed !!
        y = y + dwt_res
        
        y = self.iwt(y)
        return y + residual


