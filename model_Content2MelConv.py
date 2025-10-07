# %% VC CONVERTER: Content+F0 -> Mel (FiLM with your centroid)
import torch, torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, spk_dim, hid):
        super().__init__()
        self.lin = nn.Linear(spk_dim, 2 * hid)

    def forward(self, h, spk):
        # h: (B,T,H), spk: (B,spk_dim)
        g, b = self.lin(spk).chunk(2, dim=-1)  # (B,H),(B,H)
        return h * g.unsqueeze(1) + b.unsqueeze(1)


class ConvBlock(nn.Module):
    def __init__(self, ch, ks=5, d=1):
        super().__init__()
        pad = (ks // 2) * d
        self.conv = nn.Conv1d(ch, ch, ks, padding=pad, dilation=d)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x):  # x:(B,C,T)
        return F.gelu(self.bn(self.conv(x)))


class Content2Mel(nn.Module):
    def __init__(self, content_dim=1024, spk_dim=256, hid=384, n_mels=80, n_blocks=6):
        super().__init__()
        self.inp = nn.Linear(content_dim + 1, hid)  # +1 for F0
        self.film = FiLM(spk_dim, hid)
        self.blocks = nn.ModuleList(
            [ConvBlock(hid, ks=5, d=2 ** (i % 3)) for i in range(n_blocks)]
        )
        self.proj = nn.Linear(hid, n_mels)

    def forward(self, H, f0, spk):
        # H:(B,T,1024)  f0:(B,T)  spk:(B,256)
        x = torch.cat([H, f0.unsqueeze(-1)], dim=-1)
        h = self.inp(x)
        h = self.film(h, spk)
        h = h.transpose(1, 2)  # (B,H,T)
        for blk in self.blocks:
            h = h + blk(h)
        h = h.transpose(1, 2)  # (B,T,H)
        mel = self.proj(h)  # (B,T,80)  (log10-mel)
        return mel