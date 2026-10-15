"""FT-Transformer (Feature Tokenizer + Transformer; Gorishniy, Rubachev, Khrulkov & Babenko 2021, "Revisiting Deep Learning
Models for Tabular Data", NeurIPS) for the four-class label, with its attention map as an embedded feature selector.

Every feature becomes a token (numeric: value x learned vector + learned bias; categorical: a learned embedding per category);
a [CLS] token is prepended; Transformer blocks let tokens attend to each other; the prediction is read from the [CLS] token.
Feature importance (paper Sec. 5.4): the attention the [CLS] token pays to each feature token, averaged over heads, layers
and households -- a feature the model consults more gets more weight.

Small-data settings (300-650 households per country): 2 blocks, 32-dimensional tokens, 4 heads, dropout 0.2, AdamW,
inverse-frequency class weights in the loss, early stopping on a stratified 15% validation split (patience 20 epochs).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, x):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + a
        return x + self.ffn(self.n2(x)), w                       # w: (batch, tokens, tokens), averaged over heads


class _Net(nn.Module):
    def __init__(self, n_num, cat_sizes, d=32, blocks=2, heads=4, dropout=0.2, n_classes=4):
        super().__init__()
        self.w_num = nn.Parameter(torch.randn(n_num, d) * 0.1)
        self.b_num = nn.Parameter(torch.zeros(n_num, d))
        self.cat = nn.ModuleList([nn.Embedding(s, d) for s in cat_sizes])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([_Block(d, heads, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, n_classes))

    def forward(self, xn, xc):
        tok = [xn.unsqueeze(-1) * self.w_num + self.b_num] if xn.shape[1] else []
        tok += [emb(xc[:, j]).unsqueeze(1) for j, emb in enumerate(self.cat)]
        x = torch.cat([self.cls.expand(xn.shape[0] if xn.shape[1] else xc.shape[0], -1, -1)] + tok, dim=1)
        maps = []
        for b in self.blocks:
            x, w = b(x)
            maps.append(w[:, 0, 1:])                              # [CLS] -> feature tokens
        return self.head(x[:, 0]), torch.stack(maps).mean(0)


class FTTransformerClassifier:
    def __init__(self, cat_features: list[str], seed: int = 0, epochs: int = 200, patience: int = 20, lr: float = 5e-4,
                 batch: int = 64, **net):
        self.cat_features, self.seed, self.epochs, self.patience, self.lr, self.batch, self.net_kw = \
            cat_features, seed, epochs, patience, lr, batch, net
        self.classes_ = np.arange(4)

    def _encode(self, x: pd.DataFrame):
        num = x[self.num_].apply(pd.to_numeric, errors="coerce").fillna(self.median_)
        xn = ((num - self.mean_) / self.std_).to_numpy(np.float32)
        xc = (np.column_stack([x[c].astype(str).map(self.vocab_[c]).fillna(0).to_numpy(np.int64) for c in self.cat_])
              if self.cat_ else np.zeros((len(x), 0), np.int64))
        return torch.tensor(xn, device=DEVICE), torch.tensor(xc, device=DEVICE)

    def fit(self, x: pd.DataFrame, y, x_val: pd.DataFrame | None = None, y_val=None, sample_weight=None):
        """`sample_weight` (optional, training households only) multiplies each household's class-weighted loss."""
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        y = np.asarray(y, int)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if x_val is None:
            x, x_val, y, y_val, sw, _ = train_test_split(x, y, sw, test_size=0.15, stratify=y, random_state=self.seed)
        y_val = np.asarray(y_val, int)
        self.features_ = list(x.columns)
        self.cat_ = [c for c in self.features_ if c in self.cat_features]
        self.num_ = [c for c in self.features_ if c not in self.cat_]
        num = x[self.num_].apply(pd.to_numeric, errors="coerce")
        self.median_ = num.median().fillna(0)
        num = num.fillna(self.median_)
        self.mean_, self.std_ = num.mean(), num.std().replace(0, 1).fillna(1)
        self.vocab_ = {c: {v: i + 1 for i, v in enumerate(sorted(x[c].astype(str).unique()))} for c in self.cat_}
        self.model_ = _Net(len(self.num_), [len(self.vocab_[c]) + 1 for c in self.cat_], **self.net_kw).to(DEVICE)
        counts = np.bincount(y, minlength=4).astype(float)
        weight = torch.tensor(np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0), dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        per_hh = nn.CrossEntropyLoss(weight=weight, reduction="none")
        swt = torch.tensor(sw, dtype=torch.float32, device=DEVICE)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        xn, xc = self._encode(x); vn, vc = self._encode(x_val)
        yt, yv = torch.tensor(y, device=DEVICE), torch.tensor(y_val, device=DEVICE)
        best, best_state, wait = np.inf, None, 0
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            self.model_.train()
            for idx in torch.randperm(len(y), generator=g).split(self.batch):
                idx = idx.to(DEVICE)
                opt.zero_grad()
                wb = swt[idx] * weight[yt[idx]]
                loss = (per_hh(self.model_(xn[idx], xc[idx])[0], yt[idx]) * swt[idx]).sum() / wb.sum().clamp_min(1e-8)
                loss.backward(); opt.step()
            self.model_.eval()
            with torch.no_grad():
                v = float(loss_fn(self.model_(vn, vc)[0], yv))
            if v < best - 1e-4:
                best, wait = v, 0
                best_state = {k: t.detach().clone() for k, t in self.model_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            logits, _ = self.model_(*self._encode(x[self.features_]))
        return torch.softmax(logits, 1).cpu().numpy()

    def attention_importance(self, x: pd.DataFrame) -> pd.Series:
        """Mean [CLS]-to-feature attention over heads, layers and the given households (sums to < 1; the rest is [CLS])."""
        self.model_.eval()
        with torch.no_grad():
            _, att = self.model_(*self._encode(x[self.features_]))
        imp = att.mean(0).cpu().numpy()
        order = self.num_ + self.cat_                             # token order used in forward()
        return pd.Series(imp, index=order).reindex(self.features_)
