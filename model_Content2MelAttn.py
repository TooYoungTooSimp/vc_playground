# %% VC CONVERTER (ATTENTION): Content+F0 -> Mel with per-layer FiLM
import math, torch, torch.nn as nn
import torch.nn.functional as F


# --- sinusoidal pos enc (batch_first) ---
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=1 << 14):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: (B,T,D)
        return x + self.pe[: x.size(1)].unsqueeze(0)  # type: ignore


# --- FiLM that emits per-channel scale/bias for a given hidden dim ---
class FiLM(nn.Module):
    def __init__(self, spk_dim, hid):
        super().__init__()
        self.lin = nn.Linear(spk_dim, 2 * hid)

    def forward(self, h, spk):  # h: (B,T,H), spk: (B,S)
        g, b = self.lin(spk).chunk(2, dim=-1)  # (B,H),(B,H)
        return h * g.unsqueeze(1) + b.unsqueeze(1)


# --- a single Transformer block w/ pre-norm + FiLM in both sublayers ---
class AttnFiLMBlock(nn.Module):
    def __init__(self, d_model=384, nhead=6, ff_mult=4, dropout=0.1, spk_dim=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.film1 = FiLM(spk_dim, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.film2 = FiLM(spk_dim, d_model)
        self.do = nn.Dropout(dropout)

    def forward(self, x, spk, key_padding_mask=None):  # x:(B,T,D)
        # Self-attn sublayer
        xa = self.ln1(x)
        a, _ = self.attn(
            xa, xa, xa, key_padding_mask=key_padding_mask, need_weights=False
        )
        a = self.film1(a, spk)
        x = x + self.do(a)

        # FFN sublayer
        xf = self.ln2(x)
        f = self.ff(xf)
        f = self.film2(f, spk)
        x = x + self.do(f)
        return x


# --- Attention-based Content2Mel ---
class Content2MelAttn(nn.Module):
    """
    Inputs:
      H   : (B, T, content_dim)   # e.g., HuBERT/WavLM features
      f0  : (B, T)                # MIDI or z-scored F0 aligned to mel frames
      spk : (B, spk_dim)          # your centroid (256-d)
      lengths (optional): (B,)    # valid frame counts for masking
    Output:
      mel : (B, T, n_mels)        # log10-mel (same recipe you use)
    """

    def __init__(
        self,
        content_dim=1024,
        spk_dim=256,
        d_model=384,
        nhead=6,
        nlayers=6,
        ff_mult=4,
        n_mels=80,
        dropout=0.1,
    ):
        super().__init__()
        self.inp = nn.Linear(content_dim + 1, d_model)  # concat f0
        self.pos = PosEnc(d_model)
        self.blocks = nn.ModuleList(
            [
                AttnFiLMBlock(
                    d_model=d_model,
                    nhead=nhead,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    spk_dim=spk_dim,
                )
                for _ in range(nlayers)
            ]
        )
        self.out_ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, n_mels)

    def forward(self, H, f0, spk, lengths=None):
        # pack inputs
        x = torch.cat([H, f0.unsqueeze(-1)], dim=-1)  # (B,T, content+1)
        x = self.inp(x)
        x = self.pos(x)

        # build padding mask if lengths provided (True = pad)
        key_padding_mask = None
        if lengths is not None:
            B, T, _ = x.size()
            key_padding_mask = (
                torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
            )  # (B,T)

        # pass through attention blocks with FiLM conditioning
        for blk in self.blocks:
            x = blk(x, spk, key_padding_mask=key_padding_mask)

        x = self.out_ln(x)
        mel = self.proj(x)  # (B,T, n_mels)
        return mel
