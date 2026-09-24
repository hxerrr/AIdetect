"""滑坡标注修复工具：异常框扫描 + 分层随机重新划分

背景:
  ds1_firc 的滑坡 AP 仅 0.447（ds2 为 0.800），根因是该子集按文件编号顺序切分
  (train=firc_pic_1~99, valid=firc_pic_501~800, test=firc_pic_802~957)，
  导致 train/valid 的滑坡框尺度分布严重不一致(p50 面积 33.35% vs 18.51%)。

两种修复模式（均为非破坏性，不移动/不改写原标注）:
  scan     扫描滑坡标注异常（越界/超大/超小/异常宽高比/重复框），只报告不修改
  resplit  生成分层随机的 train/valid/test 列表文件，消除顺序切分带来的分布偏移

用法:
    python tools/fix_landslide.py --mode scan
    python tools/fix_landslide.py --mode resplit --ratios 0.8,0.13,0.07
    python tools/fix_landslide.py --mode resplit --dry-run
"""
import argparse
import collections
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from analyze_subsets import classify  # noqa: E402

NAMES = {0: "landslide", 1: "collapse", 2: "rockfall"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
OUT_DIR = ROOT / "dataset" / "splits"


def iter_images(split: str):
    d = ROOT / "dataset" / "images" / split
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def read_labels(img_path: Path, split: str):
    """返回 [(cls, cx, cy, w, h), ...]"""
    lbl = ROOT / "dataset" / "labels" / split / (img_path.stem + ".txt")
    out = []
    if not lbl.exists():
        return out
    for line in lbl.open(encoding="utf-8", errors="ignore"):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            out.append((int(parts[0]), *map(float, parts[1:5])))
        except ValueError:
            continue
    return out


def iou_xyxy(a, b):
    lt = (max(a[0], b[0]), max(a[1], b[1]))
    rb = (min(a[2], b[2]), min(a[3], b[3]))
    inter = max(0.0, rb[0] - lt[0]) * max(0.0, rb[1] - lt[1])
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def to_xyxy(b):
    _, cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def do_scan(target_cls: int, big: float, small: float, ratio: float, dup: float):
    print("=" * 84)
    print(f"{NAMES[target_cls]} 标注异常扫描（只报告，不修改任何文件）")
    print(f"阈值: 面积>{big}% 为超大, <{small}% 为超小, 宽高比>{ratio} 为异常, IoU>{dup} 视为重复")
    print("=" * 84)

    buckets = collections.Counter()
    per_subset = collections.defaultdict(collections.Counter)
    samples = collections.defaultdict(list)

    for split in ("train", "valid", "test"):
        for p in iter_images(split):
            sub = classify(p.name)
            boxes = [b for b in read_labels(p, split) if b[0] == target_cls]
            if not boxes:
                continue
            per_subset[sub]["img_with"] += 1
            per_subset[sub]["inst"] += len(boxes)
            for b in boxes:
                area = b[3] * b[4] * 100
                ar = b[3] / b[4] if b[4] > 0 else 999.0
                x1, y1, x2, y2 = to_xyxy(b)
                if area > big:
                    buckets["超大框"] += 1
                    per_subset[sub]["超大框"] += 1
                    if len(samples["超大框"]) < 6:
                        samples["超大框"].append(f"{split}/{p.name} area={area:.1f}%")
                if area < small:
                    buckets["超小框"] += 1
                    per_subset[sub]["超小框"] += 1
                if ar > ratio or ar < 1.0 / ratio:
                    buckets["异常宽高比"] += 1
                    per_subset[sub]["异常宽高比"] += 1
                if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                    buckets["越界框"] += 1
                    per_subset[sub]["越界框"] += 1
                    if len(samples["越界框"]) < 6:
                        samples["越界框"].append(f"{split}/{p.name} xyxy=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f})")
            # 同图内重复框
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if iou_xyxy(to_xyxy(boxes[i]), to_xyxy(boxes[j])) > dup:
                        buckets["重复框"] += 1
                        per_subset[sub]["重复框"] += 1
                        if len(samples["重复框"]) < 6:
                            samples["重复框"].append(f"{split}/{p.name} 框#{i}/#{j}")

    print(f"\n{'异常类型':<14}{'数量':>8}")
    print("-" * 24)
    for k, v in buckets.most_common():
        print(f"{k:<14}{v:>8}")
    if not buckets:
        print("（未发现异常）")

    print(f"\n{'子集':<14}{'含该类图':>9}{'实例':>8}{'超大':>7}{'超小':>7}{'宽高比':>8}{'越界':>7}{'重复':>7}")
    print("-" * 74)
    for sub in ("ds1_firc", "ds2_roboflow", "ds3_fromnet", "ds4_cn", "other_hash"):
        c = per_subset.get(sub)
        if not c:
            continue
        print(f"{sub:<14}{c['img_with']:>9}{c['inst']:>8}{c['超大框']:>7}{c['超小框']:>7}"
              f"{c['异常宽高比']:>8}{c['越界框']:>7}{c['重复框']:>7}")
    for k, v in samples.items():
        if v:
            print(f"\n[{k}] 样例:")
            for s in v:
                print("   ", s)


def group_key(name: str) -> str:
    """同一原始图及其全部增强变体返回相同 key。

    命名规律（已采样确认）:
      firc_pic_25.jpg      <-> firc_aug_0025_0/1/2.jpg
      g_-11_png_jpg.rf.<h> <-> g_aug_-11_png_jpg.rf.<h>
    若不按同源分组，增强图落进 valid、原图留在 train，就是数据泄漏。
    """
    stem = Path(name).stem
    n = stem.lower()
    try:
        if n.startswith("firc_aug_"):
            return "firc_" + str(int(n[len("firc_aug_"):].split("_")[0]))
        if n.startswith("firc_pic_"):
            return "firc_" + str(int(n[len("firc_pic_"):]))
    except ValueError:
        pass
    if n.startswith("g_aug_"):
        return "g_" + n[len("g_aug_"):].split(".rf.")[0]
    if n.startswith("g_"):
        return "g_" + n[len("g_"):].split(".rf.")[0]
    return "u_" + stem


def do_resplit(ratios, seed: int, dry_run: bool, out_name: str):
    r_tr, r_va, r_te = ratios
    assert abs(r_tr + r_va + r_te - 1.0) < 1e-6, "划分比例之和必须为 1"

    # 1) 按同源分组（跨原 train/valid/test 合并，避免增强图与原图被拆开）
    groups = collections.defaultdict(list)
    seen = set()
    for split in ("train", "valid", "test"):
        for p in iter_images(split):
            if p.stem in seen:
                continue
            seen.add(p.stem)
            groups[group_key(p.name)].append((p, split))

    # 2) 分层: 来源子集 x 组内是否含滑坡
    strata = collections.defaultdict(list)
    for gkey, members in groups.items():
        has = any(b[0] == 0 for p, s in members for b in read_labels(p, s))
        strata[(classify(members[0][0].name), has)].append(gkey)

    rng = random.Random(seed)
    assign = {"train": [], "valid": [], "test": []}
    print("=" * 84)
    print("分层随机重划分（按同源分组 + 子集x含滑坡 分层，杜绝增强图泄漏）")
    print("=" * 84)
    print(f"{'层(子集/含滑坡)':<28}{'组数':>7}{'train':>8}{'valid':>8}{'test':>8}")
    print("-" * 60)
    for key in sorted(strata):
        keys = strata[key][:]
        rng.shuffle(keys)
        n = len(keys)
        n_te = int(round(n * r_te))
        n_va = int(round(n * r_va))
        te, va, tr = keys[:n_te], keys[n_te:n_te + n_va], keys[n_te + n_va:]
        for k in tr:
            assign["train"] += groups[k]
        for k in va:
            assign["valid"] += groups[k]
        for k in te:
            assign["test"] += groups[k]
        print(f"{key[0] + '/' + ('有' if key[1] else '无'):<28}{n:>7}{len(tr):>8}{len(va):>8}{len(te):>8}")

    print("\n" + "=" * 84)
    print("划分后各集合的滑坡密度与实例数（应基本一致）")
    print("=" * 84)
    print(f"{'集合':<8}{'图片':>8}{'含滑坡图':>10}{'滑坡密度':>10}{'滑坡实例':>10}{'塌陷':>8}{'落石':>8}")
    print("-" * 66)
    for s in ("train", "valid", "test"):
        imgs = assign[s]
        n_with = 0
        cc = collections.Counter()
        for p, old_split in imgs:
            hit = False
            for b in read_labels(p, old_split):
                cc[b[0]] += 1
                if b[0] == 0:
                    hit = True
            n_with += int(hit)
        print(f"{s:<8}{len(imgs):>8}{n_with:>10}{n_with / max(len(imgs), 1) * 100:>9.1f}%"
              f"{cc[0]:>10}{cc[1]:>8}{cc[2]:>8}")

    if dry_run:
        print("\n[dry-run] 未写入任何文件")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in ("train", "valid", "test"):
        f = OUT_DIR / f"{out_name}_{s}.txt"
        f.write_text("\n".join(sorted(str(p.resolve()) for p, _ in assign[s])) + "\n", encoding="utf-8")
        print(f"\n[write] {len(assign[s])} 条 -> {f}")

    yaml_p = OUT_DIR / f"{out_name}.yaml"
    yaml_p.write_text(
        f"# 分层随机重划分后的数据配置（来源子集 x 是否含滑坡 双层分层）\n"
        f"# 生成: tools/fix_landslide.py --mode resplit  比例 train={r_tr} valid={r_va} test={r_te}\n"
        f"path: {(ROOT / 'dataset').resolve().as_posix()}\n"
        f"train: {(OUT_DIR / f'{out_name}_train.txt').resolve().as_posix()}\n"
        f"val: {(OUT_DIR / f'{out_name}_valid.txt').resolve().as_posix()}\n"
        f"test: {(OUT_DIR / f'{out_name}_test.txt').resolve().as_posix()}\n"
        f"nc: 3\n"
        f"names:\n  0: landslide\n  1: collapse\n  2: rockfall\n",
        encoding="utf-8",
    )
    print(f"\n[write] 数据配置 -> {yaml_p}")
    print("\n使用: python train.py --data " + str(yaml_p.relative_to(ROOT).as_posix()) + " --name yolo11s_v3")
    print("注意: 旧划分未被改动，正在跑的训练不受影响。")


def do_fix(target_cls: int, min_vis: float, drop_huge: float, dry_run: bool):
    """裁剪越界框（判定为标注/增强脚本 bug，坐标超出 [0,1] 数学上非法）

    - 越界框裁剪回 [0,1]，并据此重算 cx/cy/w/h
    - 裁剪后可见面积不足原框 min_vis 的，或完全在界外的，直接删除
    - 可选 drop_huge: 删除面积占比 >= 该值的框（整图标注，通常是误标）
    """
    stats = collections.Counter()
    per_split = collections.defaultdict(collections.Counter)
    n_files = 0
    print("=" * 84)
    print(f"{NAMES[target_cls]} 越界框修复"
          + ("（dry-run，不写盘）" if dry_run else "")
          + (f"  额外删除面积>={drop_huge}% 的框" if drop_huge else ""))
    print("=" * 84)

    for split in ("train", "valid", "test"):
        for p in iter_images(split):
            lbl = ROOT / "dataset" / "labels" / split / (p.stem + ".txt")
            if not lbl.exists():
                continue
            new_lines, modified = [], False
            for line in lbl.open(encoding="utf-8", errors="ignore"):
                raw = line.rstrip("\n")
                q = raw.split()
                if len(q) < 5:
                    new_lines.append(raw)
                    continue
                try:
                    c = int(q[0])
                    cx, cy, w, h = map(float, q[1:5])
                except ValueError:
                    new_lines.append(raw)
                    continue
                extra = q[5:]
                if c != target_cls:
                    new_lines.append(raw)
                    continue

                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                if max(-x1, -y1, x2 - 1.0, y2 - 1.0) > 0.01:
                    per_split[split]["oob"] += 1
                    nx1, ny1, nx2, ny2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
                    old_area = max((x2 - x1) * (y2 - y1), 1e-9)
                    vis = ((nx2 - nx1) * (ny2 - ny1)) / old_area
                    if nx2 - nx1 <= 1e-4 or ny2 - ny1 <= 1e-4 or vis < min_vis:
                        stats["越界_删除"] += 1
                        modified = True
                        continue
                    cx, cy, w, h = (nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1
                    stats["越界_裁剪"] += 1
                    modified = True

                if drop_huge and w * h * 100 >= drop_huge:
                    stats["超大_删除"] += 1
                    modified = True
                    continue

                new_lines.append(" ".join([str(c)] + [f"{v:.6f}" for v in (cx, cy, w, h)] + extra))

            if modified:
                n_files += 1
                if not dry_run:
                    lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    print(f"\n{'处理':<14}{'数量':>8}")
    print("-" * 24)
    for k, v in stats.most_common():
        print(f"{k:<14}{v:>8}")
    print(f"\n{'划分':<8}{'越界框':>8}")
    print("-" * 18)
    for s in ("train", "valid", "test"):
        print(f"{s:<8}{per_split[s]['oob']:>8}")
    print(f"\n涉及文件 {n_files} 个" + ("（未写盘）" if dry_run else "（已写盘）"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="scan", choices=["scan", "fix", "resplit", "backup"])
    ap.add_argument("--min-vis", type=float, default=0.2, help="裁剪后可见面积占比下限，低于则删除该框")
    ap.add_argument("--drop-huge", type=float, default=0.0, help="删除面积占比>=该值的框，0 表示不删")
    ap.add_argument("--big", type=float, default=90.0)
    ap.add_argument("--small", type=float, default=0.05)
    ap.add_argument("--ratio", type=float, default=8.0)
    ap.add_argument("--dup", type=float, default=0.9)
    ap.add_argument("--ratios", default="0.8,0.13,0.07")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default="split_v3")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.mode == "scan":
        do_scan(0, a.big, a.small, a.ratio, a.dup)
    elif a.mode == "fix":
        do_fix(0, a.min_vis, a.drop_huge, a.dry_run)
    elif a.mode == "resplit":
        do_resplit(tuple(float(x) for x in a.ratios.split(",")), a.seed, a.dry_run, a.out_name)
    else:
        src = ROOT / "dataset" / "labels"
        dst = ROOT / "dataset" / "labels_backup"
        if dst.exists():
            print(f"[skip] 备份已存在: {dst}")
            return
        shutil.copytree(src, dst)
        print(f"[backup] {src} -> {dst}")


if __name__ == "__main__":
    main()
