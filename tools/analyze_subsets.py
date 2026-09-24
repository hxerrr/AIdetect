"""按来源数据集(依文件名前缀区分)拆分统计与评估，定位拖累 mAP 的子集

背景: 本数据集由 3 个公开数据集合并而成
  ds1: firc_pic_*.jpg        落石+滑坡, 未做增强, 单独训练 mAP~0.59
  ds2: g_aug_*.jpg           落石+滑坡+塌陷, 做过增强(变换/裁剪等), 单独 mAP~0.81
  ds3: rf_from-net-*.png     落石(滑坡原未标注, 后补标), mAP~0.81

2026-09-16 口径统一(A2): ds3 的滑坡标"局部滑动块", 与 ds2 的"整个滑动块"冲突,
  已剥离 ds3 全部滑坡框(669 个)。故 ds3 的 landslide AP 为空属正常,
  landslide 水平只看 ds1_firc / ds2_roboflow / ds4_cn。

用法:
    python tools/analyze_subsets.py --mode stats
    python tools/analyze_subsets.py --mode eval --weights runs/train/yolo11s_e150b/weights/best.pt
"""
import argparse
import collections
import csv
from pathlib import Path

NAMES = {0: "landslide", 1: "collapse", 2: "rockfall"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DATASET_ROOT = Path("dataset")
OUT_DIR = Path("runs/subset_lists")


def classify(name: str) -> str:
    """按文件名前缀判定来源数据集(经文件名采样确认):
      ds1_firc       firc_*                    落石+滑坡, 未增强
      ds2_roboflow   g_*(含 g_aug 增强图与原图) 落石+滑坡+塌陷, Roboflow 导出
      ds3_fromnet    rf_*                      落石(+后补标滑坡)
      ds4_cn         cn_*                      整合筛选后重新编号的数据
    """
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


def iter_images(split: str):
    d = DATASET_ROOT / "images" / split
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def count_labels(p: Path, split: str):
    """返回 (类别计数, 等效边长列表[按640折算])"""
    lbl = DATASET_ROOT / "labels" / split / (p.stem + ".txt")
    cls_cnt = collections.Counter()
    sides = []
    if not lbl.exists():
        return cls_cnt, sides
    for line in lbl.open(encoding="utf-8", errors="ignore"):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
        except ValueError:
            continue
        cls_cnt[c] += 1
        sides.append(((w * h) ** 0.5) * 640)
    return cls_cnt, sides


def do_stats():
    print("=" * 78)
    print("各来源子集统计（图片数 / 实例数 / 目标尺度 / 格式）")
    print("=" * 78)
    grand = collections.Counter()
    cross = collections.defaultdict(collections.Counter)
    for split in ("train", "valid", "test"):
        imgs = list(iter_images(split))
        if not imgs:
            continue
        bysub = collections.defaultdict(list)
        for p in imgs:
            bysub[classify(p.name)].append(p)

        print(f"\n--- {split} (共 {len(imgs)} 张) ---")
        hdr = f"{'子集':<7}{'图片数':>8}{'占比':>8}  {'格式':<18}{'实例数':>8}{'滑坡':>7}{'塌陷':>7}{'落石':>7}{'中位边长':>10}"
        print(hdr)
        print("-" * len(hdr))
        for sub in ("ds1_firc", "ds2_roboflow", "ds3_fromnet", "ds4_cn", "other_hash"):
            ps = bysub.get(sub, [])
            if not ps:
                continue
            cls_tot = collections.Counter()
            all_sides = []
            fmts = collections.Counter()
            img_with = collections.Counter()
            for p in ps:
                c, s = count_labels(p, split)
                cls_tot.update(c)
                all_sides.extend(s)
                fmts[p.suffix.lower()] += 1
                for ci in (0, 1, 2):
                    if c[ci] > 0:
                        img_with[ci] += 1
            n_inst = sum(cls_tot.values())
            grand[f"{split}_{sub}_img"] = len(ps)
            cross[sub][split] = len(ps)
            fmt_s = ",".join(f"{k}:{v}" for k, v in fmts.most_common(2))
            med = sorted(all_sides)[len(all_sides) // 2] if all_sides else 0
            print(
                f"{sub:<7}{len(ps):>8}{len(ps)/len(imgs)*100:>7.1f}%  {fmt_s:<18}{n_inst:>8}"
                f"{cls_tot[0]:>7}{cls_tot[1]:>7}{cls_tot[2]:>7}{med:>9.1f}px"
            )
            # 含各类标注的图片占比: 用于识别"部分标注"(某类别只在极少数图上出现)
            print(
                f"{'':<7}{'含该类图片占比':<12}"
                + "".join(
                    f"{NAMES[ci]}:{img_with[ci]/len(ps)*100:>5.1f}%({img_with[ci]})" for ci in (0, 1, 2)
                )
            )
    print("\n注: 中位边长按 640 输入折算; 各子集数量占比决定其对整体 mAP 的权重")

    print("\n" + "=" * 78)
    print("跨划分一致性检查（各子集被分到 train/valid/test 的比例）")
    print("=" * 78)
    hdr = f"{'子集':<15}{'总数':>8}{'train':>16}{'valid':>16}{'test':>16}"
    print(hdr)
    print("-" * len(hdr))
    for sub in ("ds1_firc", "ds2_roboflow", "ds3_fromnet", "ds4_cn", "other_hash"):
        c = cross[sub]
        tot = sum(c.values())
        if tot == 0:
            continue
        print(
            f"{sub:<15}{tot:>8}"
            + "".join(
                f"{c[s]:>9}({c[s]/tot*100:>5.1f}%)" for s in ("train", "valid", "test")
            )
        )
    print("\n判读: 若各子集的 train/valid/test 比例差异很大(如某子集几乎全在 train),")
    print("      则 valid/test 的 mAP 不能代表模型在'该子集'上的真实水平, 存在评估偏差")


def build_lists(split: str):
    """为每个子集生成图片列表文件，返回 {子集: 列表文件路径}"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bysub = collections.defaultdict(list)
    for p in iter_images(split):
        bysub[classify(p.name)].append(str(p.resolve()))
    out = {}
    for sub, ps in bysub.items():
        f = OUT_DIR / f"{split}_{sub}.txt"
        f.write_text("\n".join(ps) + "\n", encoding="utf-8")
        out[sub] = f
        print(f"  {sub}: {len(ps)} 张 -> {f}")
    return out


def do_eval(weights: str, split: str, imgsz: int, only=None):
    from ultralytics import YOLO

    print(f"\n按子集单独评估 (split={split}, weights={weights})\n")
    lists = build_lists(split)
    model = YOLO(weights)

    rows = []
    for sub, listfile in sorted(lists.items()):
        if only and sub not in only:
            continue
        yaml_path = OUT_DIR / f"data_{split}_{sub}.yaml"
        # 必须写绝对路径, 否则 Ultralytics 会把列表文件路径相对 path 拼接导致找不到
        yaml_path.write_text(
            f"path: {DATASET_ROOT.resolve().as_posix()}\n"
            f"train: {listfile.resolve().as_posix()}\n"
            f"val: {listfile.resolve().as_posix()}\n"
            f"nc: 3\n"
            f"names:\n  0: landslide\n  1: collapse\n  2: rockfall\n",
            encoding="utf-8",
        )
        r = model.val(
            data=str(yaml_path), split="val", imgsz=imgsz,
            batch=8, workers=0, verbose=False, plots=False, device=0,
        )
        n_img = len(listfile.read_text(encoding="utf-8").splitlines())
        rows.append((sub, n_img, r.box.map50, r.box.map,
                     list(r.box.ap50), list(r.box.p), list(r.box.r)))

    print("\n" + "=" * 78)
    print("各子集 mAP 对比（同一模型、同一权重）")
    print("=" * 78)
    hdr = f"{'子集':<7}{'图片数':>8}{'mAP50':>9}{'mAP50-95':>11}   " + "".join(f"{n[:9]:>11}" for n in NAMES.values())
    print(hdr)
    print("-" * len(hdr))
    for sub, n, map50, mapv, ap50, pr, rc in rows:
        print(f"{sub:<7}{n:>8}{map50:>9.4f}{mapv:>11.4f}   " + "".join(f"{ap50[i]:>11.4f}" for i in sorted(NAMES)))
    print("\n(各列 AP50 顺序: landslide / collapse / rockfall)")

    print("\n" + "=" * 78)
    print("各类 精确率P / 召回率R（判断是漏检多还是误报多）")
    print("=" * 78)
    for sub, n, map50, mapv, ap50, pr, rc in rows:
        print(f"\n[{sub}]  {n} 张  mAP50={map50:.4f}")
        print(f"    {'类别':<12}{'P(精确)':>10}{'R(召回)':>10}{'AP50':>9}")
        for i in sorted(NAMES):
            print(f"    {NAMES[i]:<12}{pr[i]:>10.3f}{rc[i]:>10.3f}{ap50[i]:>9.3f}")
    print("\n判读: R低=漏检多(该类别训练样本不足 / 该子集风格与训练集差异大)")
    print("      P低=误报多(常见于标注不全: 图里真有目标但没标, 检测到就算错)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="stats", choices=["stats", "eval", "both"])
    ap.add_argument("--weights", default="runs/train/yolo11s_e150b/weights/best.pt")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--only", default="", help="只评估指定子集, 逗号分隔, 如 ds3_fromnet")
    a = ap.parse_args()

    if a.mode in ("stats", "both"):
        do_stats()
    if a.mode in ("eval", "both"):
        do_eval(a.weights, a.split, a.imgsz,
                only=set(x for x in a.only.split(",") if x))


if __name__ == "__main__":
    main()
