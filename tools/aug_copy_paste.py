#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线 Copy-Paste 增强（默认针对落石 rockfall, 小目标密集类别）

为什么不用内置 copy_paste:
  ultralytics 8.4.131 的 CopyPaste.apply_image() 依赖 instances.segments 填充轮廓,
  纯 bbox 数据集 segments 为空 -> 增强静默失效。故这里离线做: 裁实例 patch -> 贴到别的图。

关键设计(避免引入噪声):
  1) patch 裁剪时拒绝"带进其它目标": 若扩边后裁剪区与其它 GT 框的交集面积超过该框面积的
     --max-other-overlap(默认10%), 则弃用该候选, 防止把未标注目标贴进去造成假阴性。
  2) 粘贴位置拒绝重叠: 与"其它类别"框 IoU 需 < --iou-other(0.05); 与同类框 IoU 需
     < --iou-same(0.3)。随机试 --tries 次, 找不到就跳过这个 patch。
  3) 相对尺度保持真实: patch 按 目标图/源图 对角线比缩放, 再叠 0.85~1.2 抖动,
     并限制边长在 [16px, 0.4*短边]。
  4) 边缘羽化 + 通道颜色匹配, 减轻"贴纸感"和硬边。

安全:
  默认只出预览图(runs/aug_*), 不动数据集; --apply 才写入 dataset/images/train 与 labels/train,
  新文件名前缀 cp_ ; --clean 按清单删除; 写后清 dataset 下 *.cache。

用法:
    # 1) 先出 20 张预览, 看接缝/尺度是否合理
    python tools/aug_copy_paste.py
    # 2) 生成 2000 张写入训练集
    python tools/aug_copy_paste.py --apply --num 2000
    # 3) 反悔
    python tools/aug_copy_paste.py --clean
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path("dataset")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = {0: "landslide", 1: "collapse", 2: "rockfall"}


def label_path_for(img: Path, split: str) -> Path:
    return DATASET_ROOT / "labels" / split / (img.stem + ".txt")


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """返回 [(cls, cx, cy, w, h)] 归一化"""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            out.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        except ValueError:
            continue
    return out


def to_xyxy(b, W, H):
    _, cx, cy, w, h = b
    return np.array([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])


def inter_area(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou_xyxy(a, b) -> float:
    inter = inter_area(a, b)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build_bank(imgs: list[Path], split: str, cls: int, bank_images: int,
               min_side: int, pad: float, max_other_overlap: float,
               max_size: int, seed: int) -> list[dict]:
    """从训练图裁取目标类别 patch; 拒绝会把其它目标一起裁进来的候选"""
    rng = random.Random(seed)
    pool = imgs[:]
    rng.shuffle(pool)
    pool = pool[:bank_images]
    bank = []
    n_cand = n_rej = 0
    for img_p in pool:
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = read_boxes(label_path_for(img_p, split))
        xyxy = [to_xyxy(b, W, H) for b in boxes]
        for idx, b in enumerate(boxes):
            if b[0] != cls:
                continue
            n_cand += 1
            x1, y1, x2, y2 = xyxy[idx]
            bw, bh = x2 - x1, y2 - y1
            if max(bw, bh) < min_side:
                continue
            px, py = bw * pad, bh * pad
            cx1, cy1, cx2, cy2 = x1 - px, y1 - py, x2 + px, y2 + py
            if cx1 < 0 or cy1 < 0 or cx2 > W or cy2 > H:
                continue
            crop = np.array([cx1, cy1, cx2, cy2])
            bad = False
            for j, ob in enumerate(boxes):
                if j == idx:
                    continue
                oa = (xyxy[j][2] - xyxy[j][0]) * (xyxy[j][3] - xyxy[j][1])
                if oa > 0 and inter_area(crop, xyxy[j]) / oa > max_other_overlap:
                    bad = True
                    break
            if bad:
                n_rej += 1
                continue
            patch = im[int(cy1):int(cy2), int(cx1):int(cx2)].copy()
            if patch.size == 0:
                continue
            if max(patch.shape[:2]) > max_size:  # 超大 patch 直接跳过(显存/贴不下)
                continue
            bank.append({
                "patch": patch,
                "diag": float(np.hypot(W, H)),
                "src": img_p.name,
            })
    print(f"[i] patch 候选 {n_cand}, 因带进其它目标被拒 {n_rej}, 入池 {len(bank)}")
    return bank


def feather_mask(h: int, w: int, frac: float) -> np.ndarray:
    k = max(3, int(min(h, w) * frac))
    k = k + 1 if k % 2 == 0 else k
    m = np.ones((h, w), np.float32)
    m = cv2.GaussianBlur(m, (k, k), k / 3.0)
    return np.clip((m - m.min()) / max(1e-6, m.max() - m.min()), 0, 1)


def color_match(patch: np.ndarray, dst_region: np.ndarray) -> np.ndarray:
    """把 patch 的均值/方差对齐到目标区域, 减轻贴纸感"""
    out = patch.astype(np.float32)
    dst = dst_region.astype(np.float32)
    for c in range(3):
        p, d = out[..., c], dst[..., c]
        ps, ds = p.std() + 1e-3, d.std() + 1e-3
        out[..., c] = np.clip((p - p.mean()) * (ds / ps) + d.mean(), 0, 255)
    return out.astype(np.uint8)


def paste(im: np.ndarray, patch: np.ndarray, x1: int, y1: int, feather: float):
    H, W = im.shape[:2]
    ph, pw = patch.shape[:2]
    x2, y2 = x1 + pw, y1 + ph
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return
    sub = patch[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
    if feather > 0:
        a = feather_mask(sub.shape[0], sub.shape[1], feather)[..., None]
    else:
        a = np.ones((sub.shape[0], sub.shape[1], 1), np.float32)
    region = im[cy1:cy2, cx1:cx2]
    sub = color_match(sub, region)
    im[cy1:cy2, cx1:cx2] = (sub * a + region * (1 - a)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", type=int, default=2, help="增强的类别id, 2=落石")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num", type=int, default=2000, help="生成的新图片数")
    ap.add_argument("--per-img", type=int, default=2, help="每张新图平均粘贴个数")
    ap.add_argument("--bank-images", type=int, default=3000, help="从多少张图里采集 patch")
    ap.add_argument("--min-side", type=int, default=20, help="patch 源框最小边长(px)")
    ap.add_argument("--pad", type=float, default=0.15, help="裁剪时向外扩边比例")
    ap.add_argument("--max-other-overlap", type=float, default=0.10)
    ap.add_argument("--iou-other", type=float, default=0.05)
    ap.add_argument("--iou-same", type=float, default=0.30)
    ap.add_argument("--tries", type=int, default=30)
    ap.add_argument("--feather", type=float, default=0.12, help="边缘羽化比例, 0=关闭")
    ap.add_argument("--max-size", type=int, default=320, help="patch 最大边长(px)")
    ap.add_argument("--out", default="", help="输出目录, 默认 runs/aug_<cls>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=20, help="非 --apply 时生成多少张预览")
    ap.add_argument("--apply", action="store_true", help="写入 dataset/images/<split> 与 labels/<split>")
    ap.add_argument("--clean", action="store_true", help="按清单删除已生成的图片与标签")
    a = ap.parse_args()

    out_dir = Path(a.out) if a.out else Path("runs") / f"aug_{NAMES[a.cls]}"
    man_path = out_dir / "manifest.csv"

    if a.clean:
        if not man_path.exists():
            print(f"[!] 清单不存在: {man_path}")
            return
        n = 0
        for row in csv.DictReader(man_path.open(encoding="utf-8-sig")):
            for k in ("image", "label"):
                p = Path(row[k])
                if p.exists():
                    p.unlink()
            n += 1
        man_path.unlink()
        print(f"[OK] 已删除 {n} 组增强数据")
        for c in DATASET_ROOT.rglob("*.cache"):
            c.unlink()
        return

    img_dir = DATASET_ROOT / "images" / a.split
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        print(f"[!] 无图片: {img_dir}")
        return

    rng = random.Random(a.seed)
    print(f"[i] 类别={NAMES[a.cls]}(id={a.cls}) split={a.split} 可用图={len(imgs)}")
    bank = build_bank(imgs, a.split, a.cls, a.bank_images, a.min_side, a.pad,
                      a.max_other_overlap, a.max_size, a.seed)
    if not bank:
        print("[!] patch 池为空, 放宽 --min-side / --max-other-overlap 再试")
        return

    n_out = a.num if a.apply else a.preview
    write_dir = img_dir if a.apply else (out_dir / "preview")
    lab_dir = DATASET_ROOT / "labels" / a.split if a.apply else (out_dir / "preview_labels")
    write_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    bases = imgs[:]
    rng.shuffle(bases)
    manifest, n_pasted, n_skipped = [], 0, 0

    for i in range(n_out):
        base = bases[i % len(bases)]
        im = cv2.imread(str(base))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = read_boxes(label_path_for(base, a.split))
        xyxy = [to_xyxy(b, W, H) for b in boxes]
        k = max(1, int(rng.gauss(a.per_img, 0.8)))
        new_boxes = []
        for _ in range(k):
            src = bank[rng.randrange(len(bank))]
            patch = src["patch"]
            s = (np.hypot(W, H) / src["diag"]) * rng.uniform(0.85, 1.2)
            ph, pw = patch.shape[:2]
            nh, nw = int(round(ph * s)), int(round(pw * s))
            nh = int(np.clip(nh, 16, min(H, W) * 0.4))
            nw = int(np.clip(nw, 16, min(H, W) * 0.4))
            p = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            if rng.random() < 0.5:
                p = cv2.flip(p, 1)

            placed = False
            for _t in range(a.tries):
                x1 = rng.randrange(0, max(1, W - nw))
                y1 = rng.randrange(0, max(1, H - nh))
                nb = np.array([x1, y1, x1 + nw, y1 + nh])
                if any(iou_xyxy(nb, o) >= (a.iou_same if boxes[j][0] == a.cls else a.iou_other)
                       for j, o in enumerate(xyxy)):
                    continue
                if any(iou_xyxy(nb, o) >= a.iou_same for o in new_boxes):
                    continue
                paste(im, p, x1, y1, a.feather)
                new_boxes.append(nb)
                placed = True
                break
            n_pasted += placed
            n_skipped += not placed
            if not a.apply:  # 预览: 画框便于检查
                for nb in new_boxes:
                    cv2.rectangle(im, (int(nb[0]), int(nb[1])), (int(nb[2]), int(nb[3])), (0, 255, 0), 2)

        stem = f"cp_{base.stem}_{i:05d}"
        img_out = write_dir / f"{stem}.jpg"
        cv2.imwrite(str(img_out), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
        for nb in new_boxes:
            cx = (nb[0] + nb[2]) / 2 / W
            cy = (nb[1] + nb[3]) / 2 / H
            lines.append(f"{a.cls} {cx:.6f} {cy:.6f} {(nb[2]-nb[0])/W:.6f} {(nb[3]-nb[1])/H:.6f}")
        lab_out = lab_dir / f"{stem}.txt"
        lab_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        manifest.append({"image": str(img_out), "label": str(lab_out),
                         "base": base.name, "pasted": len(new_boxes)})

    out_dir.mkdir(parents=True, exist_ok=True)
    with man_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "label", "base", "pasted"])
        w.writeheader()
        w.writerows(manifest)

    print("\n" + "=" * 70)
    print(f"生成图片        : {len(manifest)}")
    print(f"成功粘贴/跳过   : {n_pasted} / {n_skipped}  (跳过=找不到不重叠位置)")
    print(f"平均每张新增实例: {n_pasted/max(1,len(manifest)):.2f}")
    print(f"清单            : {man_path.resolve()}")
    if a.apply:
        for c in DATASET_ROOT.rglob("*.cache"):
            c.unlink()
            print(f"  [i] 删除缓存: {c}")
        print(f"[OK] 已写入 {write_dir} 与 {lab_dir}")
        print(f"    清理: python tools/aug_copy_paste.py --clean")
    else:
        print(f"[!] 预览模式, 未写入数据集: {write_dir.resolve()}")
        print(f"    确认后: python tools/aug_copy_paste.py --apply --num {a.num}")


if __name__ == "__main__":
    main()
