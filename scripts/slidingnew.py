# ============================================================
# Colab-ready script: build windows (X,Y,A) and draw error heatmap
# ============================================================

# ---- (0) Optional: Mount Google Drive ----
# from google.colab import drive
# drive.mount('/content/drive')

# ---- (1) Imports & Settings ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os

# --- Window config (match your thesis defaults) ---
PAST, FUTURE = 10, 5       # T=10, H=5
STRIDE = 1                  # sliding stride=1
SMOOTH_K = 0                # 0=off; or use odd numbers like 3/5
ONLY_CLASS = None           # e.g., 'clipper'; or None for all

# --- I/O paths (EDIT these to your layout) ---
IN_DIR  = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/detection_results"
OUT_DIR = IN_DIR
VAL_CSV = f"{IN_DIR}/tracks_val.csv"                    # your detection/tracking CSV (validation split)
VAL_NPZ = f"{OUT_DIR}/traj_val_10to5.npz"              # output windows
PRED_FILE = None                                        # e.g., "/content/drive/MyDrive/.../preds_val.npy" (shape [N,H,2])
FIG_DIR = "/content/figures"
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

# ---- (2) Helpers ----
def moving_average(a, k):
    if k and k % 2 == 1 and k > 1:
        ker = np.ones(k) / k
        x = np.convolve(a[:, 0], ker, mode="same")
        y = np.convolve(a[:, 1], ker, mode="same")
        return np.stack([x, y], axis=-1).astype(np.float32)
    return a

def stem(s): return Path(str(s)).stem

def vid_from_name(stem_str: str) -> str:
    s = str(stem_str)
    return s.split("_")[0] if "_" in s else s

def frame_from_name(stem_str: str) -> int:
    digs = re.findall(r'\d+', str(stem_str))
    return int(digs[-1]) if digs else 0

def fill_gaps_by_t(coords, ts):
    """
    Linear interpolation inside [t_min, t_max], no extrapolation.
    coords: [N,2], ts: [N] (non-decreasing)
    returns: full_xy [T,2], full_t [T]
    """
    ts = np.asarray(ts, dtype=int)
    coords = np.asarray(coords, dtype=np.float32)

    # de-dup same frame (keep last; we will sort by conf before)
    uniq, idx = np.unique(ts, return_index=True)
    ts = uniq
    coords = coords[idx]

    t_min, t_max = ts[0], ts[-1]
    full_t = np.arange(t_min, t_max + 1, dtype=int)
    full_xy = np.empty((len(full_t), 2), dtype=np.float32)
    for j in range(2):
        full_xy[:, j] = np.interp(full_t, ts, coords[:, j])
    return full_xy, full_t

# ---- (3) Build windows: X, Y, A (anchor), meta ----
def csv_to_windows(csv_path, out_npz,
                   past=PAST, future=FUTURE, stride=STRIDE,
                   smooth_k=SMOOTH_K, only_class=ONLY_CLASS):
    df = pd.read_csv(csv_path)

    # add vid / t if missing
    if "vid" not in df.columns:
        df["video_stem"] = df["video"].apply(stem)
        df["vid"] = df["video_stem"].apply(vid_from_name)
    if "t" not in df.columns:
        if "frame" in df.columns and df["frame"].notna().any():
            df["t"] = pd.to_numeric(df["frame"], errors="coerce").fillna(0).astype(int)
        else:
            df["video_stem"] = df.get("video_stem", df["video"].apply(stem))
            df["t"] = df["video_stem"].apply(frame_from_name).astype(int)

    if only_class:
        df = df[df["cls"] == only_class]

    need = {"vid", "track_id", "cls", "cx", "cy", "t"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{csv_path} missing columns: {miss}")

    Xs, Ys, ANCH, META = [], [], [], []
    for (vid, tid, cls_name), g in df.groupby(["vid", "track_id", "cls"], dropna=False):
        # sort and dedup (keep highest conf per frame if available)
        if "conf" in g.columns:
            g = g.sort_values(["t", "conf"]).drop_duplicates(subset="t", keep="last")
        else:
            g = g.sort_values("t")

        coords = g[["cx", "cy"]].to_numpy(np.float32)
        ts = g["t"].to_numpy(int)

        # linear interpolation inside segment
        coords, ts = fill_gaps_by_t(coords, ts)
        # optional smoothing
        coords = moving_average(coords, smooth_k)

        N = len(coords)
        win = past + future
        if N < win:
            continue

        for i in range(0, N - win + 1, stride):
            x_win = coords[i:i+past]                   # [past,2]
            y_win = coords[i+past:i+past+future]       # [future,2]
            anchor = x_win[-1]                         # last observation as anchor

            Xs.append(x_win)
            Ys.append(y_win)
            ANCH.append(anchor)
            META.append((vid, int(tid), str(cls_name), int(ts[i])))

    X = np.asarray(Xs, dtype=np.float32)
    Y = np.asarray(Ys, dtype=np.float32)
    A = np.asarray(ANCH, dtype=np.float32)            # [N,2]
    META = np.asarray(META, dtype=object)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, X=X, Y=Y, A=A, meta=META,
        past=past, future=future, stride=stride,
        mode="sequential", smooth_k=smooth_k, only_class=only_class
    )
    print(f"✅ saved windows: {out_npz}\n   X: {X.shape}, Y: {Y.shape}, A: {A.shape}, windows: {len(META)}")
    return out_npz

# ---- (4) Error + Heatmap ----
def mse_window(yhat, ytrue):         # mean over horizon & dims
    return np.mean((yhat - ytrue)**2, axis=(1, 2))    # [N]

def error_heatmap(centers, err, bins=50, min_count=10, out_path=os.path.join(FIG_DIR, "error_heatmap.pdf")):
    cx, cy = centers[:,0], centers[:,1]
    cx = np.clip(cx, 0, 1); cy = np.clip(cy, 0, 1)

    H, xedges, yedges = np.histogram2d(cx, cy, bins=bins, range=[[0,1],[0,1]])
    S, _, _ = np.histogram2d(cx, cy, bins=[xedges, yedges], weights=err)
    M = np.divide(S, H, out=np.zeros_like(S), where=H>0)  # mean error per bin
    M_masked = np.ma.masked_where(H < min_count, M)

    plt.figure(figsize=(5, 4.6))
    im = plt.imshow(M_masked.T, origin="lower", extent=[0,1,0,1], aspect="equal")
    plt.xlabel("x (normalised)"); plt.ylabel("y (normalised)")
    cbar = plt.colorbar(im); cbar.set_label("mean window error")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300); plt.close()
    print(f"✅ saved heatmap: {out_path}")

# ---- (5) Run A: build validation windows ----
csv_to_windows(VAL_CSV, VAL_NPZ)

# ---- (6) Run B: load predictions (or make a naive baseline), compute error, draw heatmap ----
val = np.load(VAL_NPZ)
Y_true = val["Y"]     # [N,H,2]
A      = val["A"]     # [N,2]
N, H = Y_true.shape[0], Y_true.shape[1]

if PRED_FILE and Path(PRED_FILE).exists():
    Y_hat = np.load(PRED_FILE)
    assert Y_hat.shape == Y_true.shape, f"Prediction shape {Y_hat.shape} != Y_true {Y_true.shape}"
    print(f"✅ loaded predictions: {PRED_FILE}  shape={Y_hat.shape}")
else:
    # Fallback baseline: last observation hold (repeat anchor across horizon)
    print("⚠️ PRED_FILE not provided/found. Using naive last-observation baseline for the heatmap.")
    Y_hat = np.repeat(A[:, None, :], H, axis=1).astype(np.float32)

err_win = mse_window(Y_hat, Y_true)               # [N]
error_heatmap(A, err_win, bins=50, min_count=10, out_path=f"{FIG_DIR}/error_heatmap.pdf")

# (optional) also save a PNG for quick preview in doc editors
error_heatmap(A, err_win, bins=50, min_count=10, out_path=f"{FIG_DIR}/error_heatmap.png")
