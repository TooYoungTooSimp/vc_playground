import math

import torch
import torch.nn as nn


class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=1 << 14):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (L, D)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # type: ignore


class XVector(nn.Module):
    def __init__(
        self, feat_dim=80, d_model=256, nhead=4, nlayers=4, emb_dim=256, num_classes=100
    ):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=nlayers,
            enable_nested_tensor=False,
        )
        self.pos = PosEnc(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classify = nn.Linear(emb_dim, num_classes)

    def forward(self, x, lengths):
        # x: (B,T,80)
        x = self.pos(self.proj(x))  # (B,T,d_model)
        B, T, _ = x.size()
        mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]  # (B,T)
        rev_mask = ~mask
        h = self.encoder(x, src_key_padding_mask=mask)  # (B,T,d_model)
        # print(h.shape, mask.shape) # torch.Size([8, 697, 256]) torch.Size([8, 697])
        rev_mask = rev_mask.unsqueeze(-1)
        lengths = lengths.unsqueeze(-1)
        mean = (h * rev_mask).sum(1) / lengths
        m2 = ((h * h) * rev_mask).sum(1) / lengths
        var = (m2 - mean * mean).clamp_min(1e-6)
        std = torch.sqrt(var)
        pooled = torch.cat([mean, std], dim=1)
        emb = self.fc(pooled)  # (B, emb_dim)
        logits = self.classify(emb)  # (B, num_classes)
        return logits, emb

    def load_ckpt(self, pth):
        state = torch.load(pth)
        self.load_state_dict(state)