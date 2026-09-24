"""检查候选权重能否加载，并打印其训练 imgsz / 类别名，避免推理参数对不上

用法:
    python tools/check_models.py models/yolov8s/weights/best.pt models/yolov11s/weights/best.pt
"""
import sys

import ultralytics
from ultralytics import YOLO

print("ultralytics", ultralytics.__version__)
paths = sys.argv[1:] or [
    "models/yolov8s/weights/best.pt",
    "models/yolov11s/weights/best.pt",
    "models/yolov26s/weights/best.pt",
    "runs/detect/runs/train/yolo11s_v2-4/weights/best.pt",
]
for p in paths:
    try:
        m = YOLO(p)
        ta = (m.ckpt or {}).get("train_args") or {}
        print(f"[OK]   {p}")
        print(f"       imgsz={ta.get('imgsz')}  task={m.task}  names={m.names}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {p}\n       {str(e)[:200]}")
