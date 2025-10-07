from model_Content2MelAttn import Content2MelAttn
from model_Content2MelConv import Content2Mel
from model_XVector import XVector

# batch1 = next(iter(train_loader))
# logmel, labels, lengths = batch1
# model = XVector(num_classes=len(label_vocab))
# logits, emb = model(logmel, lengths)
# print(logits.shape, emb.shape)
# num_classes = len(label_vocab)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = XVector(
#     feat_dim=80,
#     d_model=256,
#     nhead=4,
#     nlayers=3,
#     emb_dim=256,
#     num_classes=num_classes,
# ).to(device)
# opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
# sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
# crit = nn.CrossEntropyLoss()
