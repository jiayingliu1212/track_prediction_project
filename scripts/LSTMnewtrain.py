# =========================
# Step 3: 训练 LSTM（10 -> 5）
# =========================
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt            # NEW
import pandas as pd                        # NEW
import os                                  # NEW

TR = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/detection_results/traj_train_10to5.npz"
VA = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/detection_results/traj_val_10to5.npz"
OUT_DIR = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/detection_results"  # NEW
os.makedirs(OUT_DIR, exist_ok=True)        # NEW

data_tr = np.load(TR, allow_pickle=True); Xtr, Ytr = data_tr["X"], data_tr["Y"]
data_va = np.load(VA, allow_pickle=True); Xva, Yva = data_va["X"], data_va["Y"]
print("Train:", Xtr.shape, Ytr.shape, " | Val:", Xva.shape, Yva.shape)

class TrajSet(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)  # [N,10,2]
        self.Y = torch.from_numpy(Y)  # [N, 5,2]
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

dl_tr = DataLoader(TrajSet(Xtr, Ytr), batch_size=256, shuffle=True, drop_last=False)
dl_va = DataLoader(TrajSet(Xva, Yva), batch_size=512, shuffle=False, drop_last=False)

class TrajLSTM(nn.Module):
    def __init__(self, in_dim=2, hidden=128, layers=2, out_steps=5, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=drop)
        self.fc   = nn.Linear(hidden, out_steps*2)
        self.out_steps = out_steps
    def forward(self, x):                      # x: [B,10,2]
        y,_ = self.lstm(x)                     # [B,10,H]
        last = y[:, -1]                        # [B,H]
        out  = self.fc(last).view(-1, self.out_steps, 2)  # [B,5,2]
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajLSTM().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.MSELoss()

best = float("inf")
EPOCHS = 50
save_path = f"{OUT_DIR}/lstm_10to5.pt"

# ---------- 记录历史（用于画 learning curve） ----------
hist_epochs, hist_tr, hist_va = [], [], []          # NEW

print(f"Start training on {device} ...")
for ep in range(1, EPOCHS+1):
    # train
    model.train(); tr=0.0
    for xb, yb in dl_tr:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = crit(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tr += loss.item()*len(xb)
    tr /= len(dl_tr.dataset)

    # val
    model.eval(); va=0.0
    with torch.no_grad():
        for xb, yb in dl_va:
            xb, yb = xb.to(device), yb.to(device)
            va += crit(model(xb), yb).item()*len(xb)
    va /= len(dl_va.dataset)

    print(f"epoch {ep:02d} | train {tr:.6f} | val {va:.6f}")

    # 记录
    hist_epochs.append(ep)                    # NEW
    hist_tr.append(tr)                        # NEW
    hist_va.append(va)                        # NEW

    # 简单早停/保存
    if va < best:
        best = va
        torch.save(model.state_dict(), save_path)
        print(f"  -> saved best: {best:.6f} to {save_path}")

print("Best val MSE:", best)

# ---------- 保存并绘制学习曲线 ----------
# 1) 保存 CSV（方便论文留档/复现实验）
df = pd.DataFrame({"epoch": hist_epochs, "train_mse": hist_tr, "val_mse": hist_va})  # NEW
csv_path = f"{OUT_DIR}/lstm_curves.csv"                                             # NEW
df.to_csv(csv_path, index=False)                                                     # NEW
print("Curves saved to:", csv_path)

# 2) 画图并保存（Matplotlib，论文直接插图）
def _smooth(x, k=1):                           # 可选平滑，k=1 表示不平滑
    if k <= 1: return x
    import numpy as np
    w = np.ones(k)/k
    return np.convolve(np.array(x, dtype=float), w, mode="same")

SMOOTH_K = 3  # 想要更平滑可以设 5；不想平滑设 1
plt.figure(figsize=(4.0, 3.0))
plt.plot(hist_epochs, _smooth(hist_tr, SMOOTH_K), label="train")
plt.plot(hist_epochs, _smooth(hist_va, SMOOTH_K), label="val")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
curve_png = f"{OUT_DIR}/lstm_curves.png"
plt.savefig(curve_png, dpi=200)
print("Curve figure saved to:", curve_png)
