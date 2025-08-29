import os

label_dirs = [
    "./dataset/labels/train",
    "./dataset/labels/val"
]

valid_class_ids = set(range(7))  # 类别编号应为 0~6

class_count = [0] * 7
total_files = 0
error_files = 0

for label_dir in label_dirs:
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".txt"):
            continue

        total_files += 1
        fpath = os.path.join(label_dir, fname)
        with open(fpath) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[❌格式错误] {fpath} line {i+1}: {line.strip()}")
                error_files += 1
                continue

            try:
                cls, cx, cy, w, h = map(float, parts)
                cls = int(cls)
            except:
                print(f"[❌类型错误] {fpath} line {i+1}: {line.strip()}")
                error_files += 1
                continue

            if cls not in valid_class_ids:
                print(f"[❌非法类别] {fpath} line {i+1}: 类别 {cls} 不在 0–6")
                error_files += 1

            if not all(0.0 <= v <= 1.0 for v in [cx, cy, w, h]):
                print(f"[❌越界坐标] {fpath} line {i+1}: {cx:.3f},{cy:.3f},{w:.3f},{h:.3f}")
                error_files += 1

            if 0 <= cls < 7:
                class_count[cls] += 1

print("\n✅ 检查完成！")
print(f"共检查标签文件：{total_files} 个")
print(f"发现错误标签文件：{error_files} 个\n")

for cid, count in enumerate(class_count):
    print(f"类别 {cid}: 出现 {count} 次")
