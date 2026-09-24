"""统计数据集目标尺度分布，判断瓶颈是否在"小目标数据不足"

用法:
    python tools/analyze_dataset.py                     # 默认分析 train 集
    python tools/analyze_dataset.py --labels dataset/labels/valid

尺度定义(按训练输入 640 折算后的等效边长 sqrt(w*h)):
    tiny  < 16px | small 16~32px | medium 32~96px | large >= 96px
"""
import argparse
import collections
from pathlib import Path

NAMES = {0: "landslide", 1: "collapse", 2: "rockfall"}


def bucket(side):
    if side < 16:
        return "tiny"
    if side < 32:
        return "small"
    if side < 96:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="dataset/labels/train")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    stats = collections.defaultdict(lambda: collections.Counter())
    sides = collections.defaultdict(list)
    imgs_with = collections.defaultdict(set)

    files = sorted(Path(a.labels).rglob("*.txt"))
    for f in files:
        for line in f.open(encoding="utf-8"):
            p = line.split()
            if len(p) < 5:
                continue
            try:
                c = int(p[0])
                w, h = float(p[3]) * a.imgsz, float(p[4]) * a.imgsz
            except ValueError:
                continue
            side = (w * h) ** 0.5
            stats[c][bucket(side)] += 1
            stats[c]["total"] += 1
            sides[c].append(side)
            imgs_with[c].add(f)

    print(f"labels: {a.labels}   图片数: {len(files)}   折算分辨率: {a.imgsz}\n")
    header = f"{'class':<11}{'total':>8}{'tiny<16':>10}{'small':>9}{'medium':>9}{'large':>9}   {'中位边长':>8}  {'图片数':>7}"
    print(header)
    print("-" * len(header))

    tot = collections.Counter()
    for c in sorted(stats):
        s = stats[c]
        tot.update(s)
        n = s["total"]
        ss = sorted(sides[c])
        med = ss[len(ss) // 2] if ss else 0
        print(
            f"{NAMES.get(c, str(c)):<11}{n:>8}"
            f"{s['tiny']:>6}{s['tiny']/n*100:>5.1f}%"
            f"{s['small']:>6}{s['small']/n*100:>5.1f}%"
            f"{s['medium']:>6}{s['medium']/n*100:>5.1f}%"
            f"{s['large']:>6}{s['large']/n*100:>5.1f}%"
            f"{med:>9.1f}px{len(imgs_with[c]):>8}"
        )

    n = tot["total"]
    print("-" * len(header))
    print(
        f"{'ALL':<11}{n:>8}"
        f"{tot['tiny']:>6}{tot['tiny']/n*100:>5.1f}%"
        f"{tot['small']:>6}{tot['small']/n*100:>5.1f}%"
        f"{tot['medium']:>6}{tot['medium']/n*100:>5.1f}%"
        f"{tot['large']:>6}{tot['large']/n*100:>5.1f}%"
    )

    small_ratio = (tot["tiny"] + tot["small"]) / n * 100
    print(f"\n小目标(等效边长<32px)占比: {small_ratio:.1f}%")
    print("注: 小目标占比高 -> 提输入分辨率/copy-paste 增强收益大")
    print("    小目标占比低 -> 瓶颈更可能在模型容量或标注一致性")


if __name__ == "__main__":
    main()
