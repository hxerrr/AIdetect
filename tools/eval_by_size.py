"""按目标尺度分组统计召回率，定位瓶颈是"小目标"还是"判别力"

逻辑: 用很低的置信度阈值(conf=0.01)跑预测，对每个 GT 判断是否存在
      IoU>0.5 且类别一致的预测框。若某尺寸段召回明显偏低，说明该尺寸是瓶颈。

用法:
    python tools/eval_by_size.py --weights runs/train/yolo11s_e150b/weights/best.pt
"""
import argparse
import os
from pathlib import Path

# Windows WDDM 下 PyTorch 会尝试借用共享内存，导致超出显存的异常分配，必须早于 import torch 设置
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

NAMES = {0: "landslide", 1: "collapse", 2: "rockfall"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
BUCKETS = [("tiny<16", 0, 16), ("small16-32", 16, 32), ("medium32-96", 32, 96), ("large>=96", 96, 1e9)]


def bucket_of(side):
    for name, lo, hi in BUCKETS:
        if lo <= side < hi:
            return name
    return BUCKETS[-1][0]


def load_gt(label_path):
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return np.zeros((0, 5), dtype=np.float32)
    for line in p.open(encoding="utf-8"):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        boxes.append([c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 5)


def iou_xyxy(a, b):
    """a:(N,4) b:(M,4) -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0", help="显存被其他进程抢占时改用 cpu")
    a = ap.parse_args()

    img_dir = Path("dataset/images") / a.split
    lbl_dir = Path("dataset/labels") / a.split
    files = [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in EXTS]
    print(f"split={a.split}  图片 {len(files)} 张  conf={a.conf}  IoU阈值={a.iou}")

    from ultralytics import YOLO

    model = YOLO(a.weights)
    # source 传目录(而非路径列表): 走 dataloader 才会真正按 batch 分批处理，
    # 传 list 时 Ultralytics 可能一次性 stack 成 (N,H,W,C) 导致分配十几 GB
    gen = model.predict(
        source=str(img_dir), imgsz=a.imgsz, conf=a.conf, iou=0.7,
        max_det=a.max_det, batch=a.batch, stream=True, verbose=False, device=a.device,
    )

    stat = {}
    n_done = 0
    for res in gen:
        n_done += 1
        f = Path(res.path)
        lbl = lbl_dir / (f.stem + ".txt")
        gt = load_gt(lbl)
        if len(gt) == 0:
            continue
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            pred_xyxy = np.zeros((0, 4), dtype=np.float32)
            pred_cls = np.zeros((0,), dtype=np.int64)
        else:
            # boxes.xyxy 是原图像素坐标，GT 是归一化坐标，必须归一到同一坐标系再算 IoU
            pred_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            oh, ow = res.orig_shape[:2]
            pred_xyxy[:, [0, 2]] /= ow
            pred_xyxy[:, [1, 3]] /= oh
            pred_cls = boxes.cls.cpu().numpy().astype(np.int64)

        ious = iou_xyxy(gt[:, 1:5], pred_xyxy)
        for i, g in enumerate(gt):
            c = int(g[0])
            side = float(np.sqrt((g[3] - g[1]) * (g[4] - g[2])) * a.imgsz)
            bk = bucket_of(side)
            key = (c, bk)
            hit = False
            if len(pred_cls):
                mask = pred_cls == c
                if mask.any():
                    hit = bool(ious[i][mask].max() >= a.iou)
            s = stat.setdefault(key, [0, 0])
            s[1] += 1
            s[0] += int(hit)

        if n_done % 300 == 0:
            print(f"  ...{n_done}/{len(files)}", flush=True)

    print("\n按类别 x 尺度的召回率（conf=0.01，已排除置信度阈值影响）\n")
    header = f"{'class':<11}{'尺度':<13}{'GT数':>8}{'命中':>8}{'召回':>9}"
    print(header)
    print("-" * len(header))
    for c in sorted({k[0] for k in stat}):
        for bname, _, _ in BUCKETS:
            k = (c, bname)
            if k not in stat:
                continue
            hit, tot = stat[k]
            print(f"{NAMES.get(c, str(c)):<11}{bname:<13}{tot:>8}{hit:>8}{hit/tot*100:>8.1f}%")
        tot_c = sum(stat[(c, b)][1] for b, _, _ in BUCKETS if (c, b) in stat)
        hit_c = sum(stat[(c, b)][0] for b, _, _ in BUCKETS if (c, b) in stat)
        if tot_c:
            print(f"{'':<11}{'小计':<13}{tot_c:>8}{hit_c:>8}{hit_c/tot_c*100:>8.1f}%")
        print()

    print(f"{'按尺度汇总':<24}{'GT数':>8}{'命中':>8}{'召回':>9}")
    print("-" * len(header))
    for bname, _, _ in BUCKETS:
        tot = sum(stat[(c, bname)][1] for c in {k[0] for k in stat} if (c, bname) in stat)
        hit = sum(stat[(c, bname)][0] for c in {k[0] for k in stat} if (c, bname) in stat)
        if tot:
            print(f"{bname:<24}{tot:>8}{hit:>8}{hit/tot*100:>8.1f}%")


if __name__ == "__main__":
    main()
