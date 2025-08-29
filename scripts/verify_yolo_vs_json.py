import os
import json
import glob
from tqdm import tqdm

# 配置路径
YOLO_LABEL_DIR = "./dataset/labels"  # 你的YOLO标签路径
JSON_ROOT = "/Users/liujiaying/Desktop/Msc_project"  # 原始数据根目录

# 遍历标签 split
for split in ["train", "val"]:
    total_checked = 0
    total_errors = 0

    label_files = glob.glob(f"{YOLO_LABEL_DIR}/{split}/*.txt")
    print(f"\n🔍 正在检查 {split} 标签，一共 {len(label_files)} 个")

    for txt_path in tqdm(label_files):
        fname = os.path.basename(txt_path).replace(".txt", "")
        vid, frame_id = fname.split("_")
        json_path = os.path.join(
            JSON_ROOT,
            "Training" if split == "train" else "Validation",
            vid,
            f"{vid}.json"
        )

        if not os.path.exists(json_path):
            print(f"[跳过] 缺失 JSON 文件: {json_path}")
            continue

        with open(json_path) as f:
            json_data = json.load(f)
        gt_annos = json_data["annotations"].get(str(int(frame_id)), [])
        gt_visible = [obj for obj in gt_annos if "tool_bbox" in obj and obj.get("visibility", 1) != 0]
        gt_classes = sorted([obj["instrument"] for obj in gt_visible])

        with open(txt_path) as f:
            yolo_lines = f.readlines()
        yolo_classes = sorted([int(line.strip().split()[0]) for line in yolo_lines if line.strip()])

        if len(gt_classes) != len(yolo_classes):
            print(f"[❌ 数量不一致] {fname}: GT={len(gt_classes)} vs YOLO={len(yolo_classes)}")
            total_errors += 1
        elif gt_classes != yolo_classes:
            print(f"[❌ 类别不匹配] {fname}: GT={gt_classes} vs YOLO={yolo_classes}")
            total_errors += 1

        total_checked += 1

    print(f"\n✅ {split} 标签检查完成！共检查帧：{total_checked}")
    print(f"发现错误帧：{total_errors}")
