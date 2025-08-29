# ==== 修补 tracks_*.csv：补齐 vid 与帧号 t，并写回 CSV ====
import pandas as pd
import numpy as np
import re
from pathlib import Path

IN_DIR  = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/detection_results"
TRAIN_CSV = f"{IN_DIR}/tracks_train.csv"
VAL_CSV   = f"{IN_DIR}/tracks_val.csv"

def stem(s): return Path(str(s)).stem

def parse_vid(stem_str: str) -> str:
    """
    例：VID02_006701 -> VID02 ；若没有下划线，则取整个 stem
    """
    s = str(stem_str)
    return s.split("_")[0] if "_" in s else s

def parse_frame(stem_str: str) -> int | None:
    """
    例：VID02_006701 -> 6701
    若没有数字，返回 None
    """
    digs = re.findall(r'\d+', str(stem_str))
    return int(digs[-1]) if digs else None

def fix_csv(path: str):
    df = pd.read_csv(path)
    assert "video" in df.columns, f"{path} 缺少 video 列"

    # 1) 基于文件名 stem 生成 vid / 可能的帧号
    df["video_stem"] = df["video"].apply(stem)
    df["vid"] = df["video_stem"].apply(parse_vid)
    df["t_from_name"] = df["video_stem"].apply(parse_frame)

    # 2) 统一生成时间索引 t：
    #    优先使用已有 frame 列；否则用 t_from_name；仍为空则在每个 vid 内按文件名排序后给递增索引
    if "frame" in df.columns:
        # 尝试保留原 frame 信息
        df["t"] = pd.to_numeric(df["frame"], errors="coerce")
    else:
        df["t"] = np.nan

    # 用文件名解析到的数字填补
    df.loc[df["t"].isna(), "t"] = df.loc[df["t"].isna(), "t_from_name"]

    # 仍有缺失的，在各自 vid 组内按 video_stem 排序后赋递增索引
    def fill_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("video_stem").copy()
        if g["t"].isna().any():
            # 只给缺失的行赋序号，不覆盖已有的 t
            missing_mask = g["t"].isna()
            g.loc[missing_mask, "t"] = np.arange(missing_mask.sum())
        return g

    df = df.groupby("vid", group_keys=False).apply(fill_group)
    df["t"] = df["t"].astype(int)

    # 3) 同步把 frame 列也写上，方便后续脚本兼容
    df["frame"] = df["t"]

    # 4) 清理中间列并保存（覆盖原文件）
    out_cols = ["src","video","frame","track_id","cls","conf","cx","cy","w","h","vid","t"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan  # 保底：缺列则补空列
    df = df[out_cols].sort_values(["vid","track_id","t"]).reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

    # 简单报告
    print(f"✅ 修补完成: {path}")
    print("  示例行：")
    print(df.head(5).to_string(index=False))
    print("  组大小(top5):")
    print(df.groupby(["vid","track_id"]).size().sort_values(ascending=False).head().to_string())

# 跑 train / val
fix_csv(TRAIN_CSV)
fix_csv(VAL_CSV)
