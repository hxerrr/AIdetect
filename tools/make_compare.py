"""把多个模型的结果横向拼接，方便一眼对比

- 图片: 每个源图拼成 N 宫格对比图
- 视频: 每个源视频拼成 N 宫格对比视频（按帧对齐）

用法:
    python tools/make_compare.py --out video/output --models yolov8s-v0,yolov11s-v0,yolov26s-v0,yolo11s-v2-4
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


def captioned(im, cap, cell):
    """缩放 + 顶部标题条"""
    im = cv2.resize(im, cell)
    bar = np.zeros((30, cell[0], 3), dtype=np.uint8)
    cv2.putText(bar, cap, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    return np.vstack([bar, im])


def make_grid(cells, cols):
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    while len(rows[-1]) < cols:
        rows[-1].append(np.zeros_like(rows[0][0]))
    return np.vstack([np.hstack(r) for r in rows])


def do_images(out_root: Path, models, src_dir: Path, cell, sub="images", cmp_subdir="images"):
    cmp_dir = out_root / "compare" / cmp_subdir
    cmp_dir.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    n = 0
    for s in srcs:
        cells = []
        for m in models:
            f = out_root / m / sub / f"{s.stem}_det.jpg"
            if not f.exists():  # 兼容早期命名
                f = out_root / m / sub / f"{s.stem}_motion.jpg"
            if not f.exists():
                print(f"[warn] 缺 {f}")
                continue
            im = cv2.imread(str(f))
            if im is None:
                continue
            cells.append(captioned(im, m, cell))
        if not cells:
            continue
        cv2.imwrite(str(cmp_dir / f"{s.stem}_compare.jpg"), make_grid(cells, 2))
        n += 1
    print(f"[images] 生成 {n} 张对比图 -> {cmp_dir}")


def find_video(out_root: Path, model: str, stem: str):
    """按多种命名找结果视频。

    老模型跑的是运动目标检测 -> <原名>_motion.mp4；
    直接用 ultralytics predict 出的则是 <原名>.avi，命名对不上会被当成"缺结果"跳过。
    """
    for name in (f"{stem}_motion.mp4", f"{stem}_motion.avi",
                 f"{stem}.mp4", f"{stem}.avi"):
        f = out_root / model / name
        if f.exists():
            return f
    return None


def do_videos(out_root: Path, models, src_dir: Path, cell, fps):
    cmp_dir = out_root / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in VID_EXTS)
    for s in srcs:
        paths = []
        for m in models:
            f = find_video(out_root, m, s.stem)
            if f is None:
                print(f"[warn] 缺 {out_root / m / (s.stem + '_motion.mp4')}")
                continue
            paths.append(f)
        if len(paths) < 2:
            print(f"[skip] {s.name} 可用结果不足 2 个")
            continue
        caps = [cv2.VideoCapture(str(p)) for p in paths]
        ok = all(c.isOpened() for c in caps)
        if not ok:
            print(f"[skip] {s.name} 有视频打不开")
            for c in caps:
                c.release()
            continue
        nframes = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT) or 0) for c in caps)
        labels = [p.parent.name for p in paths]
        dst = cmp_dir / f"{s.stem}_compare.mp4"
        writer = cv2.VideoWriter(
            str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (cell[0] * 2, (cell[1] + 30) * math.ceil(len(paths) / 2)),
        )
        i = 0
        while i < nframes:
            frames = []
            good = True
            for c in caps:
                ret, fr = c.read()
                if not ret:
                    good = False
                    break
                frames.append(fr)
            if not good:
                break
            cells = [captioned(f, lb, cell) for f, lb in zip(frames, labels)]
            writer.write(make_grid(cells, 2))
            i += 1
        writer.release()
        for c in caps:
            c.release()
        print(f"[video] {s.name} -> {dst} ({i} 帧)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="video/output", help="各模型结果所在的根目录")
    ap.add_argument("--models", default="yolov8s-v0,yolov11s-v0,yolov26s-v0,yolo11s-v2-4")
    ap.add_argument("--src", default="video/input")
    ap.add_argument("--img-src", default="", help="图片源目录，留空则用 <src>/images")
    ap.add_argument("--img-subdir", default="images", help="各模型结果下的图片子目录名")
    ap.add_argument("--cmp-subdir", default="images", help="拼接结果存放目录名，位于 compare/ 下")
    ap.add_argument("--cell", default="640x360", help="单格尺寸 宽x高")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--mode", default="both", choices=["images", "videos", "both"])
    a = ap.parse_args()

    w, h = (int(x) for x in a.cell.lower().split("x"))
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    out_root, src = Path(a.out), Path(a.src)
    img_src = Path(a.img_src) if a.img_src else src / "images"

    if a.mode in ("images", "both"):
        do_images(out_root, models, img_src, (w, h), a.img_subdir, a.cmp_subdir)
    if a.mode in ("videos", "both"):
        do_videos(out_root, models, src, (w, h), a.fps)


if __name__ == "__main__":
    main()
