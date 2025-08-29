import numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path

# 路径
DIR = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/detection_results"
TR  = f"{DIR}/traj_train_10to5.npz"
VA  = f"{DIR}/traj_val_10to5.npz"
CKPT= f"{DIR}/lstm_10to5.pt"
Path(DIR).mkdir(parents=True, exist_ok=True)

# 载入数据与模型
data_tr = np.load(TR, allow_pickle=True); Xtr, Ytr = data_tr["X"], data_tr["Y"]
data_va = np.load(VA, allow_pickle=True); Xva, Yva = data_va["X"], data_va["Y"]

import torch.nn as nn
class TrajLSTM(nn.Module):
    def __init__(self, in_dim=2, hidden=128, layers=2, out_steps=5, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=drop)
        self.fc   = nn.Linear(hidden, out_steps*2); self.out_steps=out_steps
    def forward(self, x):
        y,_ = self.lstm(x); last = y[:,-1]; out = self.fc(last)
        return out.view(-1, self.out_steps, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajLSTM().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# --------- 计算 RMSE / MAE / ADE / FDE ----------
with torch.no_grad():
    xb = torch.from_numpy(Xva).to(device)
    yb = torch.from_numpy(Yva).to(device)
    pr = model(xb)                       # [N,5,2]

mse  = ((pr - yb)**2).mean().item()
rmse = float(np.sqrt(mse))
mae  = (pr - yb).abs().mean().item()

# ADE: 所有未来步的平均欧式距离；FDE: 第5步的欧式距离
diff = (pr - yb)                        # [N,5,2]
dist = torch.sqrt((diff**2).sum(dim=-1))# [N,5]
ade  = dist.mean().item()
fde  = dist[:, -1].mean().item()

print(f"Val RMSE: {rmse:.6f}  | MAE: {mae:.6f}")
print(f"ADE: {ade:.6f}  | FDE: {fde:.6f}  (all normalized in [0,1])")

# --------- 可视化若干条轨迹 ----------
N_SHOW = 6
idx = np.random.choice(len(Xva), size=min(N_SHOW, len(Xva)), replace=False)
plt.figure(figsize=(8, 2.6*N_SHOW))
for i, k in enumerate(idx, start=1):
    hist = Xva[k]                              # [10,2]
    gt   = Yva[k]                              # [5,2]
    with torch.no_grad():
        prk = model(torch.from_numpy(Xva[k:k+1]).to(device)).cpu().numpy()[0]

    gt_line = np.concatenate([hist[-1:], gt], axis=0)
    pr_line = np.concatenate([hist[-1:], prk], axis=0)

    ax = plt.subplot(N_SHOW,1,i)
    ax.plot(hist[:,0], hist[:,1], marker='o', label='History (10)')
    ax.plot(gt_line[:,0], gt_line[:,1], marker='o', label='GT Future (5)')
    ax.plot(pr_line[:,0], pr_line[:,1], marker='o', label='Pred Future (5)')
    ax.invert_yaxis(); ax.grid(True); ax.legend(loc='best')

plt.tight_layout()
out_img = f"{DIR}/lstm_10to5_examples.png"
plt.savefig(out_img, dpi=150)
print("Saved figure:", out_img)
