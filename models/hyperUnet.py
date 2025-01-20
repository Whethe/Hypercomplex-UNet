import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class PHConv2D(nn.Module):
    def __init__(self, n, in_features, out_features, kernel_size, padding=0, stride=1, cuda=False):
        super(PHConv2D, self).__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.padding = padding
        self.stride = stride
        self.cuda = cuda

        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
        self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.zeros((n, self.out_features // n, self.in_features // n, kernel_size, kernel_size))))
        self.kernel_size = kernel_size

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.F)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def kronecker_product1(self, A, F):
        siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-4:-2]))
        siz2 = torch.Size(torch.tensor(F.shape[-2:]))
        res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * F.unsqueeze(-4).unsqueeze(-6)
        siz0 = res.shape[:1]
        out = res.reshape(siz0 + siz1 + siz2)
        return out

    def kronecker_product2(self):
        H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size), device=self.A.device)
        for i in range(self.n):
            kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size,
                                                              self.kernel_size)
            H = H + kron_prod
        return H

    def forward(self, input):
        weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
        return F.conv2d(input, weight=weight, bias=self.bias, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=sqrt(5))
        nn.init.kaiming_uniform_(self.F, a=sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.F)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        return self.timembedding(t)


class DownSample(nn.Module):
    def __init__(self, in_ch, ph_n):
        super().__init__()
        self.main = PHConv2D(n=ph_n, in_features=in_ch, out_features=in_ch,
                             kernel_size=3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        self.main.reset_parameters()

    def forward(self, x, temb):
        return self.main(x)


class UpSample(nn.Module):
    def __init__(self, in_ch, ph_n):
        super().__init__()
        self.main = PHConv2D(n=ph_n, in_features=in_ch, out_features=in_ch,
                             kernel_size=3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        self.main.reset_parameters()

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.main(x)


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, ph_n, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            PHConv2D(n=ph_n, in_features=in_ch, out_features=out_ch,
                     kernel_size=3, stride=1, padding=1)
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            PHConv2D(n=ph_n, in_features=out_ch, out_features=out_ch,
                     kernel_size=3, stride=1, padding=1)
        )
        if in_ch != out_ch:
            self.shortcut = PHConv2D(n=ph_n, in_features=in_ch, out_features=out_ch,
                                     kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, PHConv2D):
                module.reset_parameters()
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class PHUNet(nn.Module):
    def __init__(self, T, in_channels, out_channels, ch, ch_mult, attn, num_res_blocks, dropout, ph_n):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        assert ch % ph_n == 0, f'Base channels {ch} must be divisible by ph_n {ph_n}'

        # For MRI reconstruction: in_channels = 2 (real + imaginary)
        assert in_channels % ph_n == 0, f'Input channels {in_channels} must be divisible by ph_n {ph_n}'
        assert out_channels % ph_n == 0, f'Output channels {out_channels} must be divisible by ph_n {ph_n}'

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = PHConv2D(n=ph_n, in_features=in_channels, out_features=ch,
                             kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when downsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, ph_n=ph_n, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch, ph_n))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, ph_n, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, ph_n, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, ph_n=ph_n, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch, ph_n))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            PHConv2D(n=ph_n, in_features=now_ch, out_features=out_channels,
                     kernel_size=3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        self.head.reset_parameters()
        if isinstance(self.tail[-1], PHConv2D):
            self.tail[-1].reset_parameters()

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)

        h = self.tail(h)
        assert len(hs) == 0
        return h