import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECA(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k//2, bias=False)
    def forward(self, x):
        y = self.avg(x).transpose(1,2)
        y = torch.sigmoid(self.conv(y)).transpose(1,2)
        return x * y

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()
        pad = (k//2) * dilation
        self.dw = nn.Conv1d(in_ch, in_ch, k, padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))

class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(7,15,31)):
        super().__init__()
        klist = list(kernels)
        b = out_ch // len(klist)
        blocks = []
        for i,k in enumerate(klist):
            out = b if i < len(klist)-1 else out_ch - b*(len(klist)-1)
            blocks.append(nn.Sequential(DepthwiseSeparableConv1d(in_ch, out, k), nn.GELU()))
        self.blocks = nn.ModuleList(blocks)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
    def forward(self, x):
        x = torch.cat([b(x) for b in self.blocks], dim=1)
        return self.fuse(x)

class ChannelSelfAttention(nn.Module):
    def __init__(self, C=64, d=32, heads=4):
        super().__init__()
        self.C = C
        self.d = d
        self.heads = heads
        assert d % heads == 0
        self.dk = d // heads
        self.q = nn.Linear(1, d, bias=False)
        self.k = nn.Linear(1, d, bias=False)
        self.v = nn.Linear(1, d, bias=False)
        self.out = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        tok = x.mean(dim=2, keepdim=True)
        q = self.q(tok).view(tok.size(0), self.C, self.heads, self.dk).transpose(1,2)
        k = self.k(tok).view(tok.size(0), self.C, self.heads, self.dk).transpose(1,2)
        v = self.v(tok).view(tok.size(0), self.C, self.heads, self.dk).transpose(1,2)
        attn = (q @ k.transpose(-1,-2)) / math.sqrt(self.dk)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1,2).contiguous().view(tok.size(0), self.C, self.d)
        out = self.out(out)
        gate = torch.sigmoid(out)
        return x * gate

class TemporalAttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, x):
        a = torch.softmax(self.w(x).squeeze(-1), dim=1)
        return (x * a.unsqueeze(-1)).sum(dim=1)

class EdgeRouterNet(nn.Module):
    def __init__(self, kernels=(7,15,31), n_ch=64, Fch=96, emb_dim=64,
                 disable_csa=False, disable_eca=False, disable_ms2=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.disable_csa = bool(disable_csa)
        self.disable_eca = bool(disable_eca)
        self.disable_ms2 = bool(disable_ms2)

        self.csa = ChannelSelfAttention(C=n_ch, d=32, heads=4)
        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, Fch, 25, padding=12, bias=False),
            nn.BatchNorm1d(Fch), nn.GELU()
        )
        self.ms1 = MultiScaleBlock(Fch, Fch, kernels=kernels)
        self.ms2 = MultiScaleBlock(Fch, Fch, kernels=kernels)
        self.eca = ECA(Fch, k=5)

        self.pool_t = nn.AvgPool1d(4,4)
        self.proj = nn.Sequential(
            nn.Conv1d(Fch, emb_dim, 1, bias=False),
            nn.BatchNorm1d(emb_dim), nn.GELU()
        )
        self.tattn = TemporalAttnPool(emb_dim)

        self.emb = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, emb_dim))
        self.len_head   = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 2))
        self.short_head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 2))
        self.long_head  = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 3))

    def forward(self, x):
        if not self.disable_csa:
            x = self.csa(x)

        f = self.stem(x)
        f = self.ms1(f)
        if not self.disable_ms2:
            f = self.ms2(f)
        if not self.disable_eca:
            f = self.eca(f)

        f = self.pool_t(f)
        f = self.proj(f)
        tok = f.transpose(1,2)
        h = self.tattn(tok)
        emb = F.normalize(self.emb(h), dim=1)
        return self.len_head(h), self.short_head(h), self.long_head(h), emb

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label):
        x = F.normalize(x, dim=1)
        W = F.normalize(self.weight, dim=1)
        cosine = F.linear(x, W)
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1,1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits
