"""批量跑多个模型的逐类别 AP 评估（串行，避免显存/内存争抢）

用法:
    python tools/eval_all.py            # 默认四个模型，在 split_v3.test 上评估
    python tools/eval_all.py --split val
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# (展示名, 权重, 类别映射)  v0 是四分类(0=fallen tree,1=landslide,2=road collapse,3=stone)，
# 必须 remap 成 1,2,3 才能和当前三分类 GT(0=landslide,1=collapse,2=rockfall) 对齐。
JOBS = [
    ("yolov8s-v1", "runs/train/yolov8s_v1/weights/best.pt", ""),
    ("yolov11s-v1", "models/yolov11s_v1/best.pt", ""),
    ("yolov8s-v0", "models/yolov8s_v0/weights/best.pt", "1,2,3"),
    ("yolov11s-v0", "models/yolov11s_v0/weights/best.pt", "1,2,3"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    split_file = f"dataset/splits/split_v3_{a.split}.txt"
    for name, w, remap in JOBS:
        cmd = [
            PY, "tools/eval_per_class.py",
            "--weights", w,
            "--data", "dataset/splits/split_v3.yaml",
            "--split", a.split,
            "--split-file", split_file,
            "--name", f"{name}_{a.split}",
            "--batch", str(a.batch),
            "--device", a.device,
            "--workers", "2",
        ]
        if remap:
            cmd += ["--remap", remap]
        print("\n[run] " + " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        print(f"[exit] {name} -> {rc}", flush=True)

    print("\n=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
