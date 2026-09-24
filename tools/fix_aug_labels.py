"""修复离线增强生成的越界标签

用法:
    python tools/fix_aug_labels.py

逻辑:
    - 只处理 labels/train 下 firc_aug_*.txt
    - 将 cx,cy,w,h 裁剪到 [0,1]
    - 若裁剪后 w*h 小于阈值则删除该框
    - 空标签文件删除，同时删除对应图片
"""

import argparse
from pathlib import Path


def fix_line(line, min_area=1e-4):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
    except ValueError:
        return None
    # 裁剪到 [0,1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    if w * h < min_area:
        return None
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="dataset/labels/train")
    p.add_argument("--images", default="dataset/images/train")
    p.add_argument("--prefix", default="firc_aug_")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    label_dir = Path(args.labels)
    image_dir = Path(args.images)
    files = sorted(label_dir.glob(f"{args.prefix}*.txt"))

    fixed_files = 0
    removed_files = 0
    removed_boxes = 0
    kept_boxes = 0

    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            fixed = fix_line(line)
            if fixed is None:
                removed_boxes += 1
                continue
            kept_boxes += 1
            new_lines.append(fixed)

        if not new_lines:
            if not args.dry_run:
                f.unlink()
                img = image_dir / (f.stem + f.suffix.replace(".txt", ".jpg"))
                if img.exists():
                    img.unlink()
                for ext in (".png", ".jpeg"):
                    img = image_dir / (f.stem + ext)
                    if img.exists():
                        img.unlink()
            removed_files += 1
            continue

        fixed_files += 1
        if not args.dry_run:
            f.write_text("".join(new_lines), encoding="utf-8")

    print(f"文件: {len(files)} | 修复保留: {fixed_files} | 空文件删除: {removed_files}")
    print(f"框: 保留 {kept_boxes} | 删除 {removed_boxes}")
    if args.dry_run:
        print("(dry-run, 未实际写入)")


if __name__ == "__main__":
    main()
