import numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn

DIR = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/detection_results"
VAL = f"{DIR}/traj_val_10to5.npz"
CKPT= f"{DIR}/lstm_10to5.pt"

# load
d = np.load(VAL, allow_pickle=True)
Xva, Yva, META = d["X"], d["Y"], d["meta"]

class TrajLSTM(nn.Module):
    def __init__(self, in_dim=2, hidden=128, layers=2, out_steps=5, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=drop)
        self.fc   = nn.Linear(hidden, out_steps*2); self.out_steps=out_steps
    def forward(self, x):
        y,_ = self.lstm(x); last = y[:,-1]
        return self.fc(last).view(-1, self.out_steps, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajLSTM().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device)); model.eval()

np.random.seed(42)
N_SHOW = 6
idx = np.random.choice(len(Xva), size=min(N_SHOW, len(Xva)), replace=False)
for i,k in enumerate(idx,1):
    vid, tid, cls, t0 = META[k]
    print(f"{i}) idx={k}  cls={cls}  vid={vid}  track_id={tid}  start_t={t0}")

plt.figure(figsize=(8, 2.6*N_SHOW))
with torch.no_grad():
    for i,k in enumerate(idx,1):
        hist, gt = Xva[k], Yva[k]
        pr = model(torch.from_numpy(Xva[k:k+1]).to(device)).cpu().numpy()[0]
        gt_line  = np.concatenate([hist[-1:], gt], axis=0)
        pr_line  = np.concatenate([hist[-1:], pr], axis=0)

        ax = plt.subplot(N_SHOW,1,i)
        ax.plot(hist[:,0], hist[:,1], marker='o', label='History (10)')
        ax.plot(gt_line[:,0], gt_line[:,1], marker='o', label='GT Future (5)')
        ax.plot(pr_line[:,0], pr_line[:,1], marker='o', label='Pred Future (5)')
        ax.invert_yaxis(); ax.grid(True)
        vid, tid, cls, _ = META[k]
        ax.set_title(f"{i}) {cls} | vid={vid} | track_id={tid}")
plt.tight_layout()
out_img = f"{DIR}/labeled_examples.png"
plt.savefig(out_img, dpi=150); print("Saved ->", out_img)
