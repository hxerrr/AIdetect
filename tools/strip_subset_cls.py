#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按子集剥离指定类别的标注 —— 用于消除"同一类别、两种标注口径"的冲突

背景:
  ds2_roboflow 的 landslide 标"整个滑动块", ds3_fromnet 的 landslide 标"局部滑动块",
  两者口径冲突, 同时训练会让模型学到矛盾的框范围, 双向拉低 landslide AP。
  方案 A2: 把 ds3_fromnet 的 landslide 全部剥离, 让 landslide 的口径只由 ds1+ds2 定义。
  (ds3 全部滑坡图都同时含其它类别, 因此只需删标签行, 不需要移出图片)

安全:
  默认只预览(dry-run), 显式 --apply 才写回; 写回前自动备份原标签到 runs/strip_*/backup。
  --undo 可从备份完整还原。
  写回后会删除 dataset 下的 *.cache, 否则 Ultralytics 仍读旧缓存。

用法:
    python tools/strip_subset_cls.py --subset ds3_fromnet --cls 0          # 预览
    python tools/strip_subset_cls.py --subset ds3_fromnet --cls 0 --apply   # 执行
    python tools/strip_subset_cls.py --subset ds3_fromnet --cls 0 --undo    # 还原
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

DATASET_ROOT = Path("dataset")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = {0: "landslide", 1: "collapse", 2: "rockfall"}


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


def read_rows(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [l for l in path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]


def collect(subset: str, split: str) -> list[tuple[str, Path]]:
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


def do_undo(out_dir: Path, split: str):
    """从备份还原标签与被移出的图片"""
    bak = out_dir / "backup"
    n_lab = n_img = 0
    for sp_dir in sorted(bak.iterdir()) if bak.exists() else []:
        if not sp_dir.is_dir():
            continue
        for f in sp_dir.glob("*.txt"):
            dst = DATASET_ROOT / "labels" / sp_dir.name / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n_lab += 1
    moved = out_dir / "removed"
    if moved.exists():
        for sp_dir in sorted(moved.iterdir()):
            if not sp_dir.is_dir():
                continue
            for f in sp_dir.iterdir():
                target = DATASET_ROOT / ("images" if f.suffix.lower() in IMG_EXTS else "labels")
                (target / sp_dir.name).mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(target / sp_dir.name / f.name))
                n_img += 1 if f.suffix.lower() in IMG_EXTS else 0
    print(f"[OK] 已还原 {n_lab} 个标签文件, {n_img} 张被移出的图片")
    clean_cache()


def clean_cache():
    n = 0
    for c in DATASET_ROOT.rglob("*.cache"):
        c.unlink()
        n += 1
        print(f"  [i] 删除缓存: {c}")
    if n == 0:
        print("  [i] 未发现 *.cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="ds3_fromnet",
                    help="ds1_firc/ds2_roboflow/ds3_fromnet/ds4_cn/other_hash")
    ap.add_argument("--cls", type=int, default=0, help="要剥离的类别id, 0=滑坡 1=塌陷 2=落石")
    ap.add_argument("--split", default="all", help="train/valid/test/all")
    ap.add_argument("--out", default="", help="输出目录, 默认 runs/strip_<subset>_<cls>")
    ap.add_argument("--drop-images", action="store_true",
                    help="剥离后标签为空时, 把图片一起移出数据集(本数据集不需要)")
    ap.add_argument("--apply", action="store_true", help="写回数据集; 默认只预览")
    ap.add_argument("--undo", action="store_true", help="从备份还原(忽略 --cls/--split)")
    a = ap.parse_args()

    out_dir = Path(a.out) if a.out else Path("runs") / f"strip_{a.subset}_{NAMES[a.cls]}"
    if a.undo:
        do_undo(out_dir, a.split)
        return

    items = collect(a.subset, a.split)
    if not items:
        print(f"[!] 未找到 {a.subset} 的图片")
        return
    print(f"[i] 子集={a.subset} 待剥离类别={NAMES[a.cls]}(id={a.cls}) 图片数={len(items)}")
    print(f"[i] 模式: {'写回(--apply)' if a.apply else '预览(dry-run)'}   输出: {out_dir.resolve()}")

    before = {c: 0 for c in NAMES}
    after = {c: 0 for c in NAMES}
    n_affected = 0
    n_emptied = 0
    n_dropped_img = 0
    manifest = []

    for split, img in items:
        lp = label_path_for(img, split)
        rows = read_rows(lp)
        kept, removed = [], []
        for line in rows:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                c = int(float(parts[0]))
            except ValueError:
                continue
            before[c] = before.get(c, 0) + 1
            (removed if c == a.cls else kept).append((c, line))
        if not removed:
            after.update({c: after.get(c, 0) + v for c, v in
                          _count(kept).items()})
            continue

        n_affected += 1
        after.update({c: after.get(c, 0) + v for c, v in _count(kept).items()})
        manifest.append({
            "split": split, "image": img.name,
            "removed": len(removed), "remain": len(kept),
        })
        if not a.apply:
            if not kept:
                n_emptied += 1
            continue

        bak_dir = out_dir / "backup" / split
        bak_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lp, bak_dir / (img.stem + ".txt"))

        if not kept and a.drop_images:
            rm_dir = out_dir / "removed" / split
            rm_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), str(rm_dir / img.name))
            lp.unlink(missing_ok=True)
            n_dropped_img += 1
            n_emptied += 1
            continue

        if not kept:
            n_emptied += 1
        lp.write_text("\n".join(l for _, l in kept) + ("\n" if kept else ""), encoding="utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["split", "image", "removed", "remain"])
        w.writeheader()
        w.writerows(manifest)

    print("\n" + "=" * 70)
    print(f"受影响图片        : {n_affected} / {len(items)}")
    print(f"剥离框数          : {before[a.cls] - after[a.cls]}")
    print(f"剥离后空标签图片  : {n_emptied}" + (f" (已连同图片移出 {n_dropped_img} 张)" if a.apply and a.drop_images else ""))
    print("-" * 70)
    print(f"{'类别':<12}{'剥离前':>10}{'剥离后':>10}")
    for c in sorted(NAMES):
        print(f"{NAMES[c]:<12}{before[c]:>10}{after[c]:>10}")
    print("-" * 70)
    print(f"清单              : {(out_dir / 'manifest.csv').resolve()}")

    if a.apply:
        clean_cache()
        print(f"\n[OK] 已写回。备份: {(out_dir / 'backup').resolve()}")
        print(f"    还原命令: python tools/strip_subset_cls.py --subset {a.subset} --cls {a.cls} --undo")
    else:
        print("\n[!] 预览模式, 未修改数据集。确认后执行:")
        print(f"    python tools/strip_subset_cls.py --subset {a.subset} --cls {a.cls} --apply")


def _count(pairs) -> dict:
    d = {c: 0 for c in NAMES}
    for c, _ in pairs:
        d[c] = d.get(c, 0) + 1
    return d


if __name__ == "__main__":
    main()
