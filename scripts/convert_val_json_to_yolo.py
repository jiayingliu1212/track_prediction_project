import os
import json
from tqdm import tqdm
from PIL import Image

# 原始数据路径（每个视频一个文件夹，含 Frames 和 JSON）
splits = {
    "train": "/Users/liujiaying/Desktop/Msc_project/Training",
    "val":   "/Users/liujiaying/Desktop/Msc_project/Validation"
}

for split, root_dir in splits.items():
    save_img_dir = f"./dataset/images/{split}"
    save_label_dir = f"./dataset/labels/{split}"
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    for vid_folder in tqdm(sorted(os.listdir(root_dir)), desc=f"[{split}]"):
        vid_path = os.path.join(root_dir, vid_folder)
        if not os.path.isdir(vid_path):
            continue

        json_path = os.path.join(vid_path, f"{vid_folder}.json")
        frames_dir = os.path.join(vid_path, "Frames")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            annotations = json.load(f)["annotations"]

        for frame_id, objects in annotations.items():
            fname = f"{int(frame_id):06}.png"
            img_src = os.path.join(frames_dir, fname)
            img_dst = os.path.join(save_img_dir, f"{vid_folder}_{fname}")
            txt_dst = os.path.join(save_label_dir, f"{vid_folder}_{fname[:-4]}.txt")

            if not os.path.exists(img_src):
                continue

            # 拷贝图像（不需要获取图像尺寸）
            img = Image.open(img_src)
            img.save(img_dst)

            with open(txt_dst, "w") as f_txt:
                for obj in objects:
                    if "tool_bbox" not in obj:
                        continue
                    if obj.get("visibility", 1) == 0:
                        continue  # 可选：跳过不可见工具

                    x, y, w, h = obj["tool_bbox"]

                    # ✅ 仅格式转换，不做归一化（原始数据已经归一化）
                    cx = x + w / 2
                    cy = y + h / 2

                    # ✅ 合法性校验
                    if not all(0.0 <= v <= 1.0 for v in [cx, cy, w, h]):
                        continue

                    label = obj["instrument"]
                    if label not in range(7):
                        print(f"⚠️ 非法类别 ID: {label} in {vid_folder}_{fname}")
                        continue

                    f_txt.write(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# ✅ 清除 YOLOv8 缓存（避免加载过时标签）
os.system("rm -f dataset/labels/train.cache")
os.system("rm -f dataset/labels/val.cache")
print("✅ 转换完成，已完成归一化与合法性校验！")
