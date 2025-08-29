from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import glob

CKPT = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/runs/detect/cholec_yolov8m_refined11/weights/best.pt"

# 本地拷贝的目录，速度快很多
TRAIN_SRC = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/dataset/images/train"
VAL_SRC = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/dataset/images/val"

# 输出到 Drive 目录
OUT_DIR = "/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/detection_results"
ONLY_CLASS = None  # 只筛某一类就写 'clipper'，否则保持 None

model = YOLO(CKPT)


def export_tracks(source, out_csv, only_class=ONLY_CLASS):
    rows = []

    # 获取总文件数（假设 source 是文件夹）
    all_files = sorted(glob.glob(f"{source}/*"))
    total_files = len(all_files)
    print(f"📂 共找到 {total_files} 个文件，开始处理 {source} ...")

    for file_idx, r in enumerate(model.track(
            source=source,
            tracker="botsort.yaml",
            conf=0.25, iou=0.5,
            imgsz=832,
            device=0,
            half=True,
            workers=2,
            stream=True,
            persist=True,
            save=False,
            verbose=False
    )):
        # 实时打印进度
        if hasattr(r, "path"):
            print(f"[{file_idx + 1}/{total_files}] 正在处理: {Path(r.path).name}")

        if r.boxes is None:
            continue

        H, W = r.orig_shape
        names = r.names
        frame_idx = getattr(r, "frame_idx", None)
        video_id = Path(getattr(r, "path", "")).stem
        ids = getattr(r.boxes, "id", None)

        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            cls_name = names.get(cls_id, str(cls_id))
            if only_class and cls_name != only_class:
                continue
            if ids is None or ids[i] is None:
                continue
            tid = int(ids[i].item())
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            conf = float(r.boxes.conf[i].item())
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            ww = (x2 - x1) / W
            hh = (y2 - y1) / H
            rows.append(dict(src=Path(source).name, video=video_id, frame=frame_idx,
                             track_id=tid, cls=cls_name, conf=conf, cx=cx, cy=cy, w=ww, h=hh))

    df = pd.DataFrame(rows).sort_values(["video", "track_id", "frame"]).reset_index(drop=True)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ 保存完成: {out_csv}，共 {len(df)} 行数据")


# 分别导出 train 和 val
export_tracks(TRAIN_SRC, f"{OUT_DIR}/tracks_train.csv")
export_tracks(VAL_SRC, f"{OUT_DIR}/tracks_val.csv")
