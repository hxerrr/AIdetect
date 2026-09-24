#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型辅助标注导出工具 —— 用于补齐"部分标注"类别(典型: ds3_fromnet 的滑坡)

原理:
  用现有 best.pt 对指定子集推理, 找出
    模型预测为某类别(默认 landslide) 且 与该类已有 GT 框的 IoU < 阈值(默认 0.3)
  的预测框 —— 这些就是"图里很可能存在、但标注缺失"的目标, 即部分标注造成的假阳性来源。

产出:
  1) vis/      带框可视化图(黄=已有GT, 绿=候选待确认, 蓝=模型预测且已匹配GT)
  2) candidates.csv  候选框清单(按置信度降序, 含与所有GT的最大IoU, 便于判断是否为误检)
  3) labels/   合并后的完整标签(GT + 候选), 默认不写回原数据集
  4) index.html 缩略图预览页, 便于快速抽查

安全:
  默认绝不修改 dataset/labels, 只有显式 --apply 才写回, 且写回前自动备份。
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path("dataset")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["landslide", "collapse", "rockfall"]
# BGR
C_GT = (0, 215, 255)     # 黄: 已有标注
C_CAND = (0, 255, 0)     # 绿: 候选(疑似漏标)
C_MATCH = (255, 128, 0)  # 蓝: 模型预测且已被GT覆盖

DEFAULT_WEIGHTS = "runs/train/yolo11s_e150b/weights/best.pt"


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


def read_gt(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 5:
            rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ax2 = ax - aw / 2, ax + aw / 2
    ay1, ay2 = ay - ah / 2, ay + ah / 2
    bx1, bx2 = bx - bw / 2, bx + bw / 2
    by1, by2 = by - bh / 2, by + bh / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def collect_images(subset: str, split: str) -> list[tuple[str, Path]]:
    splits = ["train", "valid", "test"] if split == "all" else [split]
    out = []
    for sp in splits:
        d = DATASET_ROOT / "images" / sp
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS and classify(p.name) == subset:
                out.append((sp, p))
    return out


def run_infer(model, paths: list[Path], conf: float, iou: float, imgsz: int, bs: int = 32):
    """分批推理; 若批量失败(尺寸不一致等)自动回退逐张"""
    results = []
    for i in range(0, len(paths), bs):
        chunk = paths[i : i + bs]
        try:
            results.extend(
                model.predict(
                    source=[str(p) for p in chunk],
                    conf=conf, iou=iou, imgsz=imgsz,
                    device=0, verbose=False,
                )
            )
        except Exception as e:  # 回退逐张, 保证鲁棒
            print(f"    [warn] 批量推理失败({type(e).__name__}), 回退逐张")
            for p in chunk:
                results.extend(
                    model.predict(source=str(p), conf=conf, iou=iou, imgsz=imgsz,
                                  device=0, verbose=False)
                )
    return results


def draw(img: np.ndarray, gt_rows, pred_boxes, cands, s: float) -> np.ndarray:
    """gt_rows: 全部GT; pred_boxes: [(cls,cx,cy,w,h,conf,is_cand)]"""
    H, W = img.shape[:2]

    def box_px(cx, cy, bw, bh):
        return (int((cx - bw / 2) * W * s), int((cy - bh / 2) * H * s),
                int((cx + bw / 2) * W * s), int((cy + bh / 2) * H * s))

    # 已有标注(黄)
    for c, cx, cy, bw, bh in gt_rows:
        x1, y1, x2, y2 = box_px(cx, cy, bw, bh)
        cv2.rectangle(img, (x1, y1), (x2, y2), C_GT, 1)
        cv2.putText(img, NAMES[c], (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, C_GT, 1, cv2.LINE_AA)
    # 预测(绿=候选, 蓝=已匹配)
    for c, cx, cy, bw, bh, cf, is_cand in pred_boxes:
        x1, y1, x2, y2 = box_px(cx, cy, bw, bh)
        col = C_CAND if is_cand else C_MATCH
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if is_cand else 1)
        if is_cand:
            cv2.putText(img, f"{NAMES[c]} {cf:.2f}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--subset", default="ds3_fromnet",
                    help="来源子集: ds1_firc/ds2_roboflow/ds3_fromnet/ds4_cn/other_hash")
    ap.add_argument("--cls", type=int, default=0, help="要补齐的类别id, 0=滑坡 1=塌陷 2=落石")
    ap.add_argument("--split", default="all", help="train/valid/test/all")
    ap.add_argument("--conf", type=float, default=0.5, help="候选框最低置信度")
    ap.add_argument("--min-iou", type=float, default=0.3,
                    help="与同类GT的IoU低于此值才视为漏标候选")
    ap.add_argument("--limit", type=int, default=0, help="只处理前N张(0=全部), 用于先试跑")
    ap.add_argument("--out", default="runs/pseudo_label")
    ap.add_argument("--no-vis", action="store_true", help="不保存可视化图")
    ap.add_argument("--apply", action="store_true",
                    help="把候选框写回 dataset/labels(会先备份)。默认只导出不修改")
    ap.add_argument("--gt-only", action="store_true",
                    help="不推理, 只导出含该类GT的图(画GT框), 用于横向对比各数据集的标注风格")
    args = ap.parse_args()

    if args.gt_only:
        vis_root = Path(args.out) / args.subset / f"gt_only_{NAMES[args.cls]}"
        n = 0
        for split, img_path in collect_images(args.subset, args.split):
            gt_rows = read_gt(label_path_for(img_path, split))
            if not any(g[0] == args.cls for g in gt_rows):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            s = min(1.0, 1024 / max(H, W))
            if s < 1.0:
                img = cv2.resize(img, (int(W * s), int(H * s)))
            img = draw(img, gt_rows, [], [], s)
            (vis_root / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_root / split / (img_path.stem + ".jpg")), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            n += 1
            if args.limit and n >= args.limit:
                break
        print(f"[i] 导出 {n} 张含 {NAMES[args.cls]} 标注的 GT 图 -> {vis_root.resolve()}")
        return

    out_dir = Path(args.out)
    items = collect_images(args.subset, args.split)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print(f"[!] 未找到 {args.subset} 的图片")
        return

    print(f"[i] 子集={args.subset} 待补齐类别={NAMES[args.cls]} 图片数={len(items)}")
    print(f"[i] 权重: {args.weights}")
    print(f"[i] 输出: {out_dir.resolve()}")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    vis_dir = out_dir / args.subset / "vis"
    lab_dir = out_dir / args.subset / "labels"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = []
    n_img_with_cand = 0
    conf_bins = [0] * 5  # 0.5-.6 .6-.7 .7-.8 .8-.9 .9-1.0

    paths = [p for _, p in items]
    print(f"[i] 推理中... ({len(paths)} 张)")
    results = run_infer(model, paths, conf=args.conf, iou=0.7, imgsz=640)

    for (split, img_path), r in zip(items, results):
        gt_rows = read_gt(label_path_for(img_path, split))
        gt_target = [g[1:] for g in gt_rows if g[0] == args.cls]

        boxes = r.boxes
        preds = []
        if boxes is not None and len(boxes):
            xywhn = boxes.xywhn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (cx, cy, bw, bh), cf, c in zip(xywhn, confs, clss):
                preds.append((int(c), float(cx), float(cy), float(bw), float(bh), float(cf)))

        cands = []
        drawn = []
        for c, cx, cy, bw, bh, cf in preds:
            is_cand = False
            if c == args.cls and cf >= args.conf:
                max_iou_t = max([iou_xywh((cx, cy, bw, bh), g) for g in gt_target], default=0.0)
                if max_iou_t < args.min_iou:
                    is_cand = True
                    max_iou_all = max(
                        [iou_xywh((cx, cy, bw, bh), g[1:]) for g in gt_rows], default=0.0
                    )
                    cands.append((cx, cy, bw, bh, cf, max_iou_all))
                    conf_bins[max(0, min(int((cf - 0.5) / 0.1), 4))] += 1
                    rows_csv.append({
                        "split": split,
                        "image": img_path.name,
                        "cls": c,
                        "xc": round(cx, 6), "yc": round(cy, 6),
                        "w": round(bw, 6), "h": round(bh, 6),
                        "conf": round(cf, 4),
                        "iou_same_cls_gt": round(max_iou_t, 4),
                        "iou_any_gt": round(max_iou_all, 4),
                    })
            drawn.append((c, cx, cy, bw, bh, cf, is_cand))

        if cands:
            n_img_with_cand += 1

        # 写合并标签(GT + 候选)
        lines = [f"{g[0]} {g[1]:.6f} {g[2]:.6f} {g[3]:.6f} {g[4]:.6f}" for g in gt_rows]
        lines += [f"{args.cls} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} {c[3]:.6f}" for c in cands]
        (lab_dir / split).mkdir(parents=True, exist_ok=True)
        (lab_dir / split / (img_path.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        # 可视化
        if not args.no_vis and cands:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            s = min(1.0, 1024 / max(H, W))
            if s < 1.0:
                img = cv2.resize(img, (int(W * s), int(H * s)))
            img = draw(img, gt_rows, drawn, cands, s)
            (vis_dir / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_dir / split / (img_path.stem + ".jpg")), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    # CSV
    csv_path = out_dir / f"candidates_{args.subset}.csv"
    rows_csv.sort(key=lambda r: -r["conf"])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()) if rows_csv else
                           ["split", "image", "cls", "xc", "yc", "w", "h", "conf",
                            "iou_same_cls_gt", "iou_any_gt"])
        w.writeheader()
        w.writerows(rows_csv)

    print("\n" + "=" * 70)
    print(f"处理图片        : {len(items)}")
    print(f"含候选的图片    : {n_img_with_cand} ({n_img_with_cand/len(items)*100:.1f}%)")
    print(f"候选框总数      : {len(rows_csv)}")
    print("置信度分布      : " + "  ".join(
        f"[{0.5+i*0.1:.1f}-{0.6+i*0.1:.1f})={conf_bins[i]}" for i in range(5)))
    print(f"清单            : {csv_path.resolve()}")
    print(f"合并标签        : {lab_dir.resolve()}")
    if not args.no_vis:
        print(f"可视化          : {vis_dir.resolve()}")
    print("=" * 70)

    # HTML 预览
    if not args.no_vis and n_img_with_cand:
        html = ["<!doctype html><meta charset='utf-8'>",
                "<style>body{background:#111;color:#eee;font:14px sans-serif}"
                ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}"
                ".c{border:1px solid #333;padding:6px}"
                ".c img{width:100%;display:block;cursor:pointer}"
                ".c b{color:#7f7}</style>",
                f"<h3>{args.subset} 候选漏标段 (绿框) — 共 {n_img_with_cand} 张</h3>",
                "<p>黄框=已有标注, 绿框=疑似漏标(待确认)</p><div class='g'>"]
        for split, img_path in items:
            v = vis_dir / split / (img_path.stem + ".jpg")
            if v.exists():
                rel = v.relative_to(out_dir).as_posix()
                n_c = sum(1 for r in rows_csv if r["image"] == img_path.name)
                html.append(
                    f"<div class='c'><b>{img_path.name} (+{n_c})</b>"
                    f"<a href='{rel}' target='_blank'><img loading='lazy' src='{rel}'></a></div>"
                )
        html.append("</div>")
        hp = out_dir / f"preview_{args.subset}.html"
        hp.write_text("\n".join(html), encoding="utf-8")
        print(f"预览页          : {hp.resolve()}")

    if args.apply:
        bak = out_dir / "backup_labels"
        n = 0
        for split, img_path in items:
            src = lab_dir / split / (img_path.stem + ".txt")
            dst = label_path_for(img_path, split)
            if not src.exists():
                continue
            orig = read_gt(dst)
            new = read_gt(src)
            if len(new) == len(orig):
                continue  # 无新增, 跳过
            dest_bak = bak / split
            dest_bak.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, dest_bak / (img_path.stem + ".txt"))
            shutil.copy2(src, dst)
            n += 1
        print(f"\n[✓] 已写回 {n} 个标签文件, 原标签备份于: {bak.resolve()}")
    else:
        print("\n[!] 未写回原数据集(安全模式)。确认无误后执行:")
        print(f"    python tools/pseudo_label.py --subset {args.subset} --conf {args.conf} --apply")


if __name__ == "__main__":
    main()
