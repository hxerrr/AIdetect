"""静态图片检测（不做运动判定，只画检测框）

与 motion_tracker.py 的区别:
  - 不建轨迹、不判断运动，单张图独立处理
  - 每张图用自己的视频无关时间轴，因此不会出现 "Moving/Static" 这种误导性标注
  - 颜色按"类别名"映射，不同代模型(类别编号不同)横向对比时同类同色

用法:
    python tools/detect_static.py --weights models/yolov8s/weights/best.pt \
        --source video/input/images --output video/output/yolov8s-v0/images --device cpu
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracker_common import color_for  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="图片文件或目录")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--suffix", default="_det")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.weights)
    names = getattr(model, "names", {}) or {}

    src = Path(a.source)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"[err] 没有找到图片: {a.source}")

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_box = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[warn] 读图失败: {f}")
            continue
        res = model.predict(img, imgsz=a.imgsz, conf=a.conf, iou=a.iou, device=a.device, verbose=False)[0]
        vis = img.copy()
        boxes = res.boxes
        if boxes is not None:
            for b in boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                label = names.get(cls, str(cls))
                color = color_for(label)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis, f"{label} {conf:.2f}", (x1, max(y1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )
                n_box += 1
        cv2.imwrite(str(out_dir / f"{f.stem}{a.suffix}.jpg"), vis)
        print(f"[ok] {f.name} -> {out_dir / (f.stem + a.suffix + '.jpg')}  ({len(boxes) if boxes is not None else 0} 框)")

    print(f"done: {len(files)} 张图，共 {n_box} 个检测框 -> {out_dir}")


if __name__ == "__main__":
    main()
