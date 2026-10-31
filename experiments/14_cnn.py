"""CNN baseline with early stopping on validation layers; the test fold is never used for selection."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys

import numpy as np, pandas as pd, torch, torch.nn as nn, warnings, json
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
torch.set_num_threads(4)

X = np.load('data/cache/thermal.npy')[:, ::3, ::3]
df = pd.read_parquet('data/cache/labels.parquet')
y = df['Porosity Label'].values.astype(np.float32); g = df['Layer'].values

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                            nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(s, x): return s.f(x).squeeze(1)

def train_predict(tr, te, seed, epochs=30):
    layers = np.unique(g[tr])
    n_val = max(3, int(round(0.15 * len(layers))))
    val_layers = layers[-n_val:]
    fit_idx = tr[~np.isin(g[tr], val_layers)]
    val_idx = tr[np.isin(g[tr], val_layers)]
    if (y[fit_idx] == 1).sum() < 2 or (y[val_idx] == 1).sum() < 1:
        return None

    torch.manual_seed(seed)
    t = lambda a: torch.tensor(a[:, None], dtype=torch.float32) / 1000.
    Xf, yf = t(X[fit_idx]), torch.tensor(y[fit_idx])
    Xv, yv = t(X[val_idx]), y[val_idx]
    Xt = t(X[te])
    net = Net()
    pw = torch.tensor((yf == 0).sum() / max((yf == 1).sum(), 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xf, yf), batch_size=32, shuffle=True)
    best_ap, best_state = -1, None
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); lf(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv)).numpy()
        ap = average_precision_score(yv, pv)
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xt)).numpy()

out = {}
for name, splits in (
    ('group5',  list(GroupKFold(n_splits=5).split(X, y, g))),
    ('forward', [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in (20, 30, 40, 50)]),
):
    aps, aucs = [], []
    for k, (tr, te) in enumerate(splits):
        p = train_predict(tr, te, seed=k)
        if p is None:
            print(f"{name} fold {k}: skipped (too few positives)"); continue
        aps.append(average_precision_score(y[te], p)); aucs.append(roc_auc_score(y[te], p))
        print(f"{name} fold {k}: PR={aps[-1]:.3f} ROC={aucs[-1]:.3f}", flush=True)
    out[name] = dict(pr=float(np.mean(aps)), pr_sd=float(np.std(aps)), roc=float(np.mean(aucs)), folds=aps)
    print(f"{name}: PR-AUC {np.mean(aps):.3f}±{np.std(aps):.3f}\n", flush=True)

json.dump(out, open('results/results_cnn_corrected.json', 'w'), indent=1)
print('saved results_cnn_corrected.json')
