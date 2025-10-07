import csv
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import SpeechT5HifiGan

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

from torch.utils.data import Dataset


class LogMelDataset(Dataset):
    def __init__(self, logmel, label):
        self.logmel = logmel
        self.label = label

    def __len__(self):
        return len(self.logmel)

    def __getitem__(self, idx):
        return self.logmel[idx], self.label[idx]


dataset = torch.load("dataset_slim.pt", weights_only=False)
label_vocab = sorted(set(dataset.label))
label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


def collate_fn(batch):
    logmel, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in logmel])
    feats_padded = pad_sequence([*logmel], batch_first=True)  # (B,T,F)
    labels = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)
    return feats_padded, labels, lengths


train_ds, val_ds = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(
    train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, drop_last=True
)
val_loader = DataLoader(
    val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, drop_last=True
)


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


# batch1 = next(iter(train_loader))
# logmel, labels, lengths = batch1
# model = XVector(num_classes=len(label_vocab))
# logits, emb = model(logmel, lengths)
# print(logits.shape, emb.shape)
num_classes = len(label_vocab)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = XVector(
    feat_dim=80,
    d_model=256,
    nhead=4,
    nlayers=3,
    emb_dim=256,
    num_classes=num_classes,
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
crit = nn.CrossEntropyLoss()

from tqdm.auto import tqdm
from collections import deque

for epoch in range(100):
    model.train()
    total, seen = 0.0, 0
    losses = deque(maxlen=100)
    for feats, labs, lengths in (pb := tqdm(train_loader, leave=False)):
        feats, labs, lengths = feats.to(device), labs.to(device), lengths.to(device)
        logits, emb = model(feats, lengths)
        loss = crit(logits, labs)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item() * feats.size(0)
        seen += feats.size(0)
        losses.append(loss.item())
        pb.set_postfix(loss=np.mean(losses))
    sched.step()
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for feats, labs, lengths in tqdm(val_loader, leave=False):
            feats, labs, lengths = feats.to(device), labs.to(device), lengths.to(device)
            logits, emb = model(feats, lengths)
            preds = logits.argmax(dim=1)
            total += labs.size(0)
            correct += (preds == labs).sum().item()
        acc = correct / total
        print(f"Epoch {epoch+1} Validation accuracy: {acc*100:.2f}%")

torch.save(model.state_dict(), "xvector_easy.pth")
