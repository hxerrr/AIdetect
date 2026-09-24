"""四模型对比：yolov8s-v0 / yolov8s-v1 / yolov11s-v0 / yolov11s-v1

- 视频：跑运动目标检测 -> video/output/<模型>/<原名>_motion.mp4
- 图片：纯静态检测     -> video/output/<模型>/images/<原名>_det.jpg
- 拼接：2x2 对比图/视频 -> video/output/compare/

已存在的产物会自动跳过，只有缺失的才补跑（CPU 推理较慢）。

用法:
    python tools/compare_models.py                 # 补跑缺失 + 拼接
    python tools/compare_models.py --force         # 全部重跑
    python tools/compare_models.py --only compare  # 只拼接
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
IN = ROOT / "video" / "input"
OUT = ROOT / "video" / "output"

MODELS = [
    ("yolov8s-v0", "models/yolov8s_v0/weights/best.pt"),
    ("yolov8s-v1", "runs/train/yolov8s_v1/weights/best.pt"),
    ("yolov11s-v0", "models/yolov11s_v0/weights/best.pt"),
    ("yolov11s-v1", "models/yolov11s_v1/best.pt"),
    # v2: 在修复后的 split_v3 上训练(split_v3.test 对它完全没见过，是干净评估)
    ("yolov11s-v2", "runs/train/yolo11s_v2/weights/best.pt"),
]


def run(cmd):
    print("[run] " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(ROOT), check=False).returncode


def need_video(name, v):
    return not (OUT / name / f"{v.stem}_motion.mp4").exists()


def need_images(name, sub="images"):
    d = OUT / name / sub
    return not d.exists() or not any(d.glob("*_det.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "videos", "images", "compare"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models", default="", help="只处理指定模型，逗号分隔，如 yolov8s-v1")
    ap.add_argument("--img-src", default="", help="图片源目录，留空则用 video/input/images")
    ap.add_argument("--img-subdir", default="images", help="各模型结果下的图片子目录名")
    ap.add_argument("--device", default="cpu", help="推理设备；GPU 空闲时用 0 可比 CPU 快 5-6 倍")
    a = ap.parse_args()

    IMG_SRC = Path(a.img_src) if a.img_src else (IN / "images")

    sel = [m.strip() for m in a.models.split(",") if m.strip()]
    models = MODELS if not sel else [m for m in MODELS if m[0] in sel]

    videos = sorted(IN.glob("*.mp4"))

    if a.only in ("all", "videos"):
        for name, w in models:
            todo = [v for v in videos if a.force or need_video(name, v)]
            if not todo:
                print(f"[skip] {name} 视频已齐全", flush=True)
                continue
            (OUT / name).mkdir(parents=True, exist_ok=True)
            print(f"\n========== VIDEO {name} ({len(todo)} 个) ==========", flush=True)
            for v in todo:
                print(f"--- {name} / {v.name}", flush=True)
                run([PY, "tools/motion_tracker.py", "--weights", w, "--source", v,
                     "--device", a.device, "--output", OUT / name / f"{v.stem}_motion.mp4"])

    if a.only in ("all", "images"):
        for name, w in models:
            # 必须按本次的图片子目录(test_images)判断，不能用默认的 images/：
            # images/ 里存的是上一次 video/input 对比的 6 张结果，会导致所有模型被误判为"已齐全"而跳过。
            if not a.force and not need_images(name, a.img_subdir):
                print(f"[skip] {name} 图片已齐全", flush=True)
                continue
            print(f"\n========== IMAGE {name} ==========", flush=True)
            run([PY, "tools/detect_static.py", "--weights", w, "--source", IMG_SRC,
                 "--device", a.device, "--output", OUT / name / a.img_subdir])

    # 注意: 必须包含 "images"。原写法只在 all/compare 时才跑拼接，
    # 导致 --only images 时下面的 a.only == "images" 分支永远进不去(死代码)，拼接被静默跳过。
    if a.only in ("all", "compare", "images"):
        print("\n========== COMPARE ==========", flush=True)
        cmp_models = ",".join(m[0] for m in MODELS)
        if a.only == "images":
            # 只跑了图片：只拼接图片，避免误用视频源
            run([PY, "tools/make_compare.py", "--out", OUT, "--mode", "images",
                 "--img-src", IMG_SRC, "--img-subdir", a.img_subdir, "--models", cmp_models])
        else:
            run([PY, "tools/make_compare.py", "--out", OUT, "--src", IN, "--models", cmp_models])

    print("\n=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
