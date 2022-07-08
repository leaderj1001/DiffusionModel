import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


# reference, thank you.
# https://github.com/w86763777/pytorch-ddpm


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    def __init__(self, in_dims, h, w):
        super(PositionalEncoding, self).__init__()
        self.positionalEncoding = torch.zeros((h * w, in_dims))

        for pos in range(0, h * w):
            for i in range(0, in_dims // 2):
                self.positionalEncoding[pos, 2 * i] = math.sin(pos / math.pow(10000, 2 * i / in_dims))
                self.positionalEncoding[pos, 2 * i + 1] = math.cos(pos / math.pow(10000, 2 * i / in_dims))

        self.register_buffer('positional_encoding', self.positionalEncoding)

    def forward(self, x):
        batch, C, h, w = x.size()
        out = x + self.positionalEncoding.permute(1, 0).view(1, C, h, w).to(x)
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_dims):
        super(SelfAttention, self).__init__()
        self.dk = in_dims ** 0.5

        self.q = nn.Conv2d(in_dims, in_dims, kernel_size=3, padding=1)
        self.k = nn.Conv2d(in_dims, in_dims, kernel_size=3, padding=1)
        self.v = nn.Conv2d(in_dims, in_dims, kernel_size=3, padding=1)

        self.out = nn.Conv2d(in_dims, in_dims, kernel_size=1)

    def forward(self, x):
        batch, c, h, w = x.size()
        q, k, v = self.q(x), self.k(x), self.v(x)

        q = q.view(batch, c, -1).permute(0, 2, 1)
        k = k.view(batch, c, -1)
        v = v.view(batch, c, -1)

        energy = torch.bmm(q, k) / self.dk
        attn = F.softmax(energy, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch, -1, h, w)

        out = self.out(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_dims, out_dims, residual=True, wide=True):
        super(ConvBlock, self).__init__()
        self.residual = residual
        if wide:
            mid_dims = int(out_dims * 1.)
        else:
            mid_dims = out_dims
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, mid_dims, kernel_size=3, padding=1),
            nn.GroupNorm(2, mid_dims),
            nn.GELU(),
            nn.Conv2d(mid_dims, out_dims, kernel_size=3, padding=1),
            nn.GroupNorm(2, out_dims)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv(x))
        else:
            return F.gelu(self.conv(x))


class DownConv(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(DownConv, self).__init__()

        self.max_pool = nn.MaxPool2d(2)
        self.down_conv = nn.Sequential(
            ConvBlock(in_dims, in_dims),
            ConvBlock(in_dims, out_dims, residual=False)
        )

    def forward(self, x):
        out = self.max_pool(x)
        out = self.down_conv(out)

        return out


class UpConv(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(UpConv, self).__init__()

        self.up_conv = nn.Sequential(
            ConvBlock(in_dims, in_dims),
            ConvBlock(in_dims, out_dims, residual=False)
        )

    def forward(self, x):
        out = self.up_conv(x)

        return out


class Model(nn.Module):
    def __init__(self, model_type='ddim'):
        super(Model, self).__init__()
        self.model_type = model_type

        self.stem = ConvBlock(3, 64, residual=False) # 32x32
        self.down1 = DownConv(64, 128) # 16x16
        self.down_pos1 = PositionalEncoding(128, 16, 16)
        self.down_sa1 = SelfAttention(128)
        self.down2 = DownConv(128, 256) # 8x8
        self.down_pos2 = PositionalEncoding(256, 8, 8)
        self.down3 = DownConv(256, 512) # 4x4
        self.down_pos3 = PositionalEncoding(512, 4, 4)

        self.up1 = UpConv(512 + 256, 256) # 8x8
        self.up_pos1 = PositionalEncoding(256, 8, 8)
        self.up2 = UpConv(256 + 128, 128) # 16x16
        self.up_pos2 = PositionalEncoding(128, 16, 16)
        self.up_sa1 = SelfAttention(128)
        self.up3 = UpConv(128 + 64, 64) # 32x32
        self.up_pos3 = PositionalEncoding(64, 32, 32)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.beta, self.alpha, self.sqrt_alpha, self.minus_sqrt_alpha, self.alpha_bar,\
            self.sqrt_alpha_bar, self.minus_sqrt_alpha_bar, self.alpha_prev, self.posterior_variance, self.posterior_log_variance_clipped, self.posterior_mean_coef1, self.posterior_mean_coef2 = self._gen_beta()

        self.loss = nn.MSELoss()

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1)
        )

    def forward(self, x):
        stem = self.stem(x)
        out1 = self.down1(stem)
        out1 = self.down_pos1(out1)

        out2 = self.down2(out1)
        out2 = self.down_pos2(out2)

        out3 = self.down3(out2)
        out3 = self.down_pos3(out3)

        out = torch.cat([self.upsample(out3), out2], dim=1)
        out = self.up1(out)
        out = self.up_pos1(out)

        out = torch.cat([self.upsample(out), out1], dim=1)
        out = self.up2(out)
        out = self.up_pos2(out)

        out = torch.cat([self.upsample(out), stem], dim=1)
        out = self.up3(out)
        out = self.up_pos3(out)

        out = self.out_conv(out)

        return out

    def _gen_beta(self):
        beta = torch.linspace(0.0001 ** 0.5, 0.02 ** 0.5, steps=1000).float() ** 2.
        alpha = 1 - beta
        sqrt_alpha = torch.sqrt(alpha)
        minus_sqrt_alpha = torch.sqrt(1 - alpha)
        alpha_bar = torch.cumprod(alpha, 0)
        alpha_prev = torch.tensor(np.append(1., alpha_bar.numpy()[:-1]))
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        minus_alpha_bar = 1 - alpha_bar
        minus_sqrt_alpha_bar = torch.sqrt(minus_alpha_bar)
        self.posterior = torch.tensor(0)
 
        posterior_variance = (1 - self.posterior) * beta * (1. - alpha_prev) / (1. - alpha_bar) + self.posterior * beta
        posterior_log_variance_clipped = torch.log(torch.maximum(posterior_variance, torch.tensor(1e-20)))
        posterior_mean_coef1 = beta * torch.sqrt(alpha_prev) / (1. - alpha_bar)
        posterior_mean_coef2 = (1. - alpha_prev) * torch.sqrt(alpha) / (1. - alpha_bar)

        return beta, alpha, sqrt_alpha, minus_sqrt_alpha, alpha_bar, sqrt_alpha_bar, minus_sqrt_alpha_bar, alpha_prev, posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2

    def _gen_loss(self, x):
        batch, h, w = x.size(0), x.size(2), x.size(3)
        x_0 = torch.randn([batch, 3, h, w]).to(x)

        inds = torch.randint(0, 1000, size=[batch])
        t_sqrt_alpha_bar = torch.gather(self.sqrt_alpha_bar, 0, inds).view(-1, 1, 1, 1)
        t_minus_sqrt_alpha_bar = torch.gather(self.minus_sqrt_alpha_bar, 0, inds).view(-1, 1, 1, 1)
        x_input = (t_sqrt_alpha_bar.to(x) * x) + (t_minus_sqrt_alpha_bar.to(x) * x_0.to(x))
        out = self.forward(x_input)

        loss = self.loss(out, x_0)

        return loss

    def _infer(self, x_t, t):
        with torch.no_grad():
            out = self.forward(x_t)

            if self.model_type == 'ddpm':
                if t > 1:
                    z = torch.randn([x_t.size(0), 3, 32, 32]).to(x_t)
                else:
                    z = 0

                a = 1 / torch.sqrt(self.alpha[t])
                b = (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])
                c = torch.sqrt(self.beta[t]) * z

                out = a * (x_t - b * out) + c
            else:
                sigma = 0.
                epsilon = torch.randn([x_t.size(0), 3, 32, 32]).to(x_t)
                predicted_x_0 = torch.sqrt(self.alpha_bar[t - 1]) * ((x_t - torch.sqrt(1 - self.alpha_bar[t]) * out) / torch.sqrt(self.alpha_bar[t]))
                direction_point = torch.sqrt(1 - self.alpha_bar[t - 1] - (sigma * sigma)) * out
                random_noise = sigma * epsilon

                out = predicted_x_0 + direction_point + random_noise
            return out


if __name__ == '__main__':
    # model = Model()
    # x = torch.randn([2, 3, 32, 32])

    # sa = SelfAttention(3)
    # sa(x)

    # loss = model._gen_loss(x)
    # print(loss)
    pass
