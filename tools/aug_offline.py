#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线整图增强：对指定子集做「几何变换 + 裁剪缩放 + 色彩抖动」扩增

为什么不用 copy-paste 贴落石（aug_copy_paste.py）：
  把落石 patch 贴到别的图上，落石该在山坡/岩壁上，却可能被贴到天空、马路、
  房顶上，位置完全不符合物理场景，等于往训练集里灌"假样本 + 错位置"，反而
  教坏模型。本工具改为「整图变换」增强：不新增任何目标，标签由坐标同步精确
  变换得到，零位置噪声。

为什么先对 ds1 做（数据集3 暂不增强）：
  ds1(firc_*) 未增强、单独训练 mAP~0.59，明显低于 ds2(~0.81，已做变换/裁剪增强)。
  ds1 只有 landslide + rockfall 两类，对它整图增强可同时扩充这两类的样本量，
  正好补上最弱的短板。增强图文件名保留 firc 前缀，analyze_subsets.classify()
  仍会把它归入 ds1_firc，统计口径不变。

增强手段（针对 640 输入的小目标检测，收益大且零标注风险）：
  1) 90° 整数倍旋转 k∈{0,1,2,3}：俯视/遥感图无固定朝向，方向不变性收益大
  2) 水平 / 垂直镜像（各 p=0.5）
  3) 随机裁剪缩放（zoom-in，窗口 70%~100% 再放大回原尺寸）：等价多尺度，
     小目标相对变大，直接缓解"小目标样本不足"
  4) HSV 抖动 + 亮度/对比度 + 偶发高斯模糊/噪声：光照与传感器差异鲁棒
  所有几何操作都同步作用于 bbox；90° 倍数保证框仍然紧致；裁剪后按「剩余可见
  面积比」过滤只剩边角的框（< --min-keep），避免产生残缺标注。

安全：
  · 默认只出预览（runs/aug_<subset>/preview），不动数据集；--apply 才写入
  · --apply 只写 images/<split> 与 labels/<split>，新名前缀固定为子集前缀 + "_aug_"
  · --clean 按 manifest 原样删除；写回后自动清 dataset 下 *.cache（否则读旧缓存）
  · valid/test 默认不动，保证评估口径干净

用法：
    python tools/aug_offline.py                              # 预览 20 张
    python tools/aug_offline.py --apply --factor 4           # ds1 train 扩到 4 倍
    python tools/aug_offline.py --subset ds1_firc --apply --factor 4
    python tools/aug_offline.py --clean                      # 反悔
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
# 子集 -> 增强图文件名前缀（必须让 analyze_subsets.classify() 仍能识别为同一子集）
PREFIX = {
    "ds1_firc": "firc",
    "ds2_roboflow": "g_",
    "ds3_fromnet": "rf_",
    "ds4_cn": "cn_",
}


def classify(name: str) -> str:
    n = name.lower()
    if n.startswith("firc"):
        return "ds1_firc"
    if n.startswith("g_"):
        return "ds2_roboflow"
    if n.startswith("rf_"):
        return "ds3_fromnet"
    if n.startswith("cn_"):
        return "ds4_cn"
    return "other_hash"


def label_path_for(img: Path, split: str) -> Path:
    return DATASET_ROOT / "labels" / split / (img.stem + ".txt")


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """返回 [(cls, cx, cy, w, h)]（归一化）"""
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


def to_pixel(boxes, W, H):
    """[(cls,cx,cy,w,h)] -> [[cls,x1,y1,x2,y2]]（像素）"""
    out = []
    for c, cx, cy, w, h in boxes:
        out.append([c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return out


# ----------------------------- 几何变换（标签同步） -----------------------------

def rotate90(img, boxes, times):
    """逆时针 90° × times；点映射 (x,y) -> (y, w-1-x)，w 为旋转前宽度"""
    for _ in range(times):
        h, w = img.shape[:2]
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        nb = []
        for b in boxes:
            _, x1, y1, x2, y2 = b
            xs = [w - 1 - x1, w - 1 - x2]
            ys = [y1, y2]
            nb.append([b[0], min(ys), min(xs), max(ys), max(xs)])
        boxes = nb
    return img, boxes


def flip(img, boxes, horizontal: bool):
    h, w = img.shape[:2]
    img = cv2.flip(img, 1 if horizontal else 0)
    nb = []
    for b in boxes:
        _, x1, y1, x2, y2 = b
        if horizontal:
            nb.append([b[0], w - 1 - x2, y1, w - 1 - x1, y2])
        else:
            nb.append([b[0], x1, h - 1 - y2, x2, h - 1 - y1])
    return img, nb


def crop_zoom(img, boxes, fx, fy, keep_thr):
    """裁 [fx,fy] 比例大小的窗口再放大回原尺寸（zoom-in，等价多尺度）"""
    H, W = img.shape[:2]
    cw, ch = max(8, int(round(W * fx))), max(8, int(round(H * fy)))
    cw, ch = min(cw, W), min(ch, H)
    x1 = random.randint(0, W - cw)
    y1 = random.randint(0, H - ch)
    sub = img[y1:y1 + ch, x1:x1 + cw]
    out = cv2.resize(sub, (W, H), interpolation=cv2.INTER_LINEAR)
    sx, sy = W / cw, H / ch
    nb = []
    for b in boxes:
        _, bx1, by1, bx2, by2 = b
        a0 = max(1e-6, (bx2 - bx1) * (by2 - by1))
        nx1 = min(max((bx1 - x1) * sx, 0.0), W)
        ny1 = min(max((by1 - y1) * sy, 0.0), H)
        nx2 = min(max((bx2 - x1) * sx, 0.0), W)
        ny2 = min(max((by2 - y1) * sy, 0.0), H)
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:
            continue
        if (nx2 - nx1) * (ny2 - ny1) / a0 < keep_thr:
            continue
        nb.append([b[0], nx1, ny1, nx2, ny2])
    return out, nb


# --------------------------------- 色彩变换 ---------------------------------

def photometric(img, rng, cfg):
    out = img
    if rng.random() < cfg.p_hsv:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue, cfg.hue)) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*cfg.sat), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(*cfg.val), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < cfg.p_bc:
        out = cv2.convertScaleAbs(out, alpha=rng.uniform(*cfg.contrast),
                                  beta=rng.uniform(*cfg.bright))
    if rng.random() < cfg.p_blur:
        k = rng.choice((3, 5))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < cfg.p_noise:
        sigma = rng.uniform(3.0, 10.0)
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def make_variant(img, boxes, rng, cfg, times):
    """一次完整增强：旋转 -> 镜像 -> 裁剪缩放 -> 色彩"""
    img, boxes = rotate90(img, boxes, times)
    if rng.random() < cfg.p_hflip:
        img, boxes = flip(img, boxes, True)
    if rng.random() < cfg.p_vflip:
        img, boxes = flip(img, boxes, False)
    if cfg.zoom_min < 1.0:
        img, boxes = crop_zoom(img, boxes,
                               rng.uniform(cfg.zoom_min, 1.0),
                               rng.uniform(cfg.zoom_min, 1.0),
                               cfg.min_keep)
    img = photometric(img, rng, cfg)
    return img, boxes


# ---------------------------------- 主流程 ----------------------------------

class Cfg:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="ds1_firc",
                    help="要增强的子集: ds1_firc/ds2_roboflow/ds3_fromnet/ds4_cn")
    ap.add_argument("--split", default="train", help="train/valid/test")
    ap.add_argument("--factor", type=float, default=4.0,
                    help="目标倍数: 4 = 原图 + 3 张增强图（按原图数量计）")
    ap.add_argument("--zoom-min", type=float, default=0.7, help="裁剪窗口最小占比, 1.0=关闭裁剪")
    ap.add_argument("--min-keep", type=float, default=0.25,
                    help="框裁剪后剩余面积比低于此值则丢弃该框")
    ap.add_argument("--p-hflip", type=float, default=0.5)
    ap.add_argument("--p-vflip", type=float, default=0.5)
    ap.add_argument("--p-hsv", type=float, default=0.7)
    ap.add_argument("--hue", type=float, default=10.0)
    ap.add_argument("--sat", type=float, nargs=2, default=[0.7, 1.4])
    ap.add_argument("--val", type=float, nargs=2, default=[0.7, 1.4])
    ap.add_argument("--p-bc", type=float, default=0.5)
    ap.add_argument("--contrast", type=float, nargs=2, default=[0.8, 1.25])
    ap.add_argument("--bright", type=float, nargs=2, default=[-20, 20])
    ap.add_argument("--p-blur", type=float, default=0.15)
    ap.add_argument("--p-noise", type=float, default=0.15)
    ap.add_argument("--skip-empty", type=int, default=1, help="1=跳过无标注图(默认)")
    ap.add_argument("--out", default="", help="输出目录, 默认 runs/aug_<subset>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=20, help="非 --apply 时预览张数")
    ap.add_argument("--apply", action="store_true", help="写入 dataset")
    ap.add_argument("--clean", action="store_true", help="按清单删除已生成的增强数据")
    a = ap.parse_args()

    out_dir = Path(a.out) if a.out else Path("runs") / f"aug_{a.subset}"
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
    src = [p for p in sorted(img_dir.iterdir())
           if p.suffix.lower() in IMG_EXTS and classify(p.name) == a.subset]
    if not src:
        print(f"[!] {img_dir} 下没有 {a.subset} 的图片")
        return

    n_var = max(0, int(round(a.factor)) - 1)
    if n_var <= 0:
        print("[!] factor 需 > 1")
        return

    cfg = Cfg()
    for k, v in vars(a).items():
        setattr(cfg, k, v)

    rng = random.Random(a.seed)
    print(f"[i] 子集={a.subset} split={a.split} 原图={len(src)} 张, 每张生成 {n_var} 张 -> 目标 {a.factor}x")
    print(f"[i] 预览={not a.apply}  输出={out_dir.resolve()}")

    write_dir = img_dir if a.apply else (out_dir / "preview")
    lab_dir = DATASET_ROOT / "labels" / a.split if a.apply else (out_dir / "preview_labels")
    write_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    prefix = PREFIX.get(a.subset, "aug_")
    manifest = []
    n_used = n_out = n_skip_empty = n_box_in = n_box_out = 0

    for si, p in enumerate(src):
        boxes = read_boxes(label_path_for(p, a.split))
        n_box_in += len(boxes)
        if a.skip_empty and not boxes:
            n_skip_empty += 1
            continue
        im = cv2.imread(str(p))
        if im is None:
            print(f"  [!] 读图失败, 跳过: {p.name}")
            continue
        H, W = im.shape[:2]
        px = to_pixel(boxes, W, H)
        # 保证每张增强图朝向互不相同（4 个 90° 朝向轮转）
        order = list(range(4))
        rng.shuffle(order)
        n_used += 1

        for j in range(n_var):
            if not a.apply and n_out >= a.preview:
                break
            img2, bx2 = make_variant(im.copy(), [b[:] for b in px], rng, cfg, order[j % 4])
            if not bx2:
                continue
            stem = f"{prefix}_aug_{si:04d}_{j}"
            img_out = write_dir / f"{stem}.jpg"
            lab_out = lab_dir / f"{stem}.txt"
            lines = []
            for b in bx2:
                c, x1, y1, x2, y2 = b
                lines.append(f"{c} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            if not a.apply:  # 预览: 画框，肉眼确认框是否贴合目标
                for c, x1, y1, x2, y2 in bx2:
                    cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0) if c == 0 else (0, 165, 255), 2)
            cv2.imwrite(str(img_out), img2, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            lab_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            manifest.append({"image": str(img_out), "label": str(lab_out),
                             "base": p.name, "variant": j})
            n_out += 1
            n_box_out += len(bx2)
            if not a.apply and n_out >= a.preview:
                break
        if not a.apply and n_out >= a.preview:
            break
        if a.apply and (n_used % 100 == 0):
            print(f"  已处理 {n_used}/{len(src)} 张原图 ...")

    out_dir.mkdir(parents=True, exist_ok=True)
    with man_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "label", "base", "variant"])
        w.writeheader()
        w.writerows(manifest)

    print("\n" + "=" * 70)
    print(f"参与增强原图    : {n_used} / {len(src)}" + (f"  (无标注跳过 {n_skip_empty})" if n_skip_empty else ""))
    print(f"生成增强图      : {n_out}")
    print(f"框数(单图平均): {n_box_in/max(1,n_used):.2f} -> {n_box_out/max(1,n_out):.2f}"
          f"  (裁剪后只剩边角的框会被丢弃, 属正常损耗)")
    print(f"清单            : {man_path.resolve()}")
    if a.apply:
        for c in DATASET_ROOT.rglob("*.cache"):
            c.unlink()
            print(f"  [i] 删除缓存: {c}")
        print(f"[OK] 已写入 {write_dir} 与 {lab_dir}")
        print(f"    反悔: python tools/aug_offline.py --subset {a.subset} --clean")
    else:
        print(f"[!] 预览模式, 未写入数据集: {write_dir.resolve()}")
        print(f"    确认后: python tools/aug_offline.py --subset {a.subset} --apply --factor {a.factor:g}")
    print("=" * 70)


if __name__ == "__main__":
    main()
