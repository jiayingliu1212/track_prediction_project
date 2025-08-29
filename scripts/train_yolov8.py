from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data="/content/drive/MyDrive/track_prediction_project/track_prediction_project/scripts/dataset/cholec_oversampled.yaml",
    epochs=140, imgsz=1024, batch=16, device=0, workers=8,
    mosaic=1.0, mixup=0.15, copy_paste=0.3, cutmix=0.0,
    auto_augment="randaugment", hsv_h=0.015, hsv_s=0.40, hsv_v=0.20,
    fliplr=0.5, erasing=0.10, close_mosaic=10,
    optimizer="AdamW", lr0=0.0025, lrf=0.05, cos_lr=True,
    weight_decay=0.0008, warmup_epochs=3, warmup_momentum=0.8,
    freeze=0, patience=30, seed=42, rect=False,
    name="cholec_yolov8m_refined11", project="runs/detect", resume=False
)