"""滑坡(landslide)专项诊断：标注几何一致性 + 子集剔除对照实验

背景: 整体 mAP 里 landslide 明显低于 collapse/rockfall，需要判断是
      (a) 标注口径/几何分布在 train 与 valid 之间不一致，还是
      (b) valid 里某些来源子集系统性缺少 landslide 标注，导致模型正确预测也被记成误检。

用法:
    # 1) 纯 CPU：各子集 landslide 标注几何统计（安全，训练期间可跑）
    python tools/diag_landslide.py --mode geom

    # 2) GPU：在"剔除某子集"后的 valid 上重算 AP，与全量 AP 对照
    python tools/diag_landslide.py --mode eval --weights runs/train/xxx/weights/best.pt --exclude ds3_fromnet

判读:
    geom 看 train/valid 的框面积分布是否可比；
    eval 若剔除 ds3 后 landslide AP 大幅上升，说明 AP 被"该子集没有滑坡标注"系统性压低。
"""
import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from analyze_subsets import classify, iter_images  # noqa: E402

NAMES = {0: "landslide", 1: "collapse", 2: "rockfall"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
OUT_DIR = ROOT / "runs" / "subset_lists"


def quantiles(vals, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not vals:
        return [0.0] * len(qs)
    s = sorted(vals)
    return [s[min(int(q * len(s)), len(s) - 1)] for q in qs]


def do_geom(target_cls: int):
    print("=" * 92)
    print(f"{NAMES[target_cls]} 标注几何统计（面积占比 = w*h，已按整图归一化；单位 %）")
    print("=" * 92)
    hdr = (f"{'split':<7}{'子集':<14}{'图片':>7}{'含该类':>8}{'实例':>8}"
           f"{'面积p10':>9}{'p50':>8}{'p90':>8}{'宽p50':>8}{'高p50':>8}{'宽高比p50':>10}")
    print(hdr)
    print("-" * len(hdr))
    for split in ("train", "valid"):
        bysub = collections.defaultdict(list)
        for p in iter_images(split):
            bysub[classify(p.name)].append(p)
        for sub in ("ds1_firc", "ds2_roboflow", "ds3_fromnet", "ds4_cn", "other_hash"):
            ps = bysub.get(sub, [])
            if not ps:
                continue
            areas, ws, hs, ratios, n_inst, n_img_with = [], [], [], [], 0, 0
            for p in ps:
                lbl = ROOT / "dataset" / "labels" / split / (p.stem + ".txt")
                if not lbl.exists():
                    continue
                hit = False
                for line in lbl.open(encoding="utf-8", errors="ignore"):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        c, w, h = int(parts[0]), float(parts[3]), float(parts[4])
                    except ValueError:
                        continue
                    if c != target_cls:
                        continue
                    hit = True
                    n_inst += 1
                    areas.append(w * h * 100)
                    ws.append(w * 100)
                    hs.append(h * 100)
                    ratios.append(w / h if h > 0 else 0.0)
                n_img_with += int(hit)
            qa, qw, qh, qr = quantiles(areas), quantiles(ws), quantiles(hs), quantiles(ratios)
            print(f"{split:<7}{sub:<14}{len(ps):>7}{n_img_with:>8}{n_inst:>8}"
                  f"{qa[0]:>9.2f}{qa[2]:>8.2f}{qa[4]:>8.2f}{qw[2]:>8.2f}{qh[2]:>8.2f}{qr[2]:>10.2f}")
        print()
    print("判读: 同一子集的 train/valid 行应接近。若 valid 的 p50 面积明显偏离 train，")
    print("      说明验证集目标尺度与训练集不一致，AP 会被分布偏移拖累。")


def build_list(split: str, only: set, exclude: set):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = []
    for p in iter_images(split):
        sub = classify(p.name)
        if only and sub not in only:
            continue
        if exclude and sub in exclude:
            continue
        keep.append(str(p.resolve()))
    tag = ("only-" + "_".join(sorted(only))) if only else ("ex-" + "_".join(sorted(exclude)) if exclude else "all")
    f = OUT_DIR / f"{split}_{tag}.txt"
    f.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"[list] {len(keep)} 张 -> {f}")
    return f, tag, len(keep)


def do_eval(weights, split, only, exclude, imgsz, batch, device):
    from ultralytics import YOLO

    listfile, tag, n = build_list(split, only, exclude)
    yaml_path = OUT_DIR / f"data_{split}_{tag}.yaml"
    yaml_path.write_text(
        f"path: {(ROOT / 'dataset').resolve().as_posix()}\n"
        f"train: {listfile.resolve().as_posix()}\n"
        f"val: {listfile.resolve().as_posix()}\n"
        f"nc: 3\n"
        f"names:\n  0: landslide\n  1: collapse\n  2: rockfall\n",
        encoding="utf-8",
    )
    r = YOLO(weights).val(
        data=str(yaml_path), split="val", imgsz=imgsz, batch=batch,
        workers=0, device=device, plots=False, verbose=False,
    )
    print("\n" + "=" * 74)
    print(f"子集对照: {tag}  ({n} 张)")
    print("=" * 74)
    print(f"{'类别':<12}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>11}")
    print("-" * 74)
    idx = list(getattr(r.box, "ap_class_index", []))
    for i, ci in enumerate(idx):
        nm = NAMES.get(int(ci), str(ci))
        try:
            p, rc, a50, a95 = r.box.p[i], r.box.r[i], r.box.ap50[i], r.box.ap[i]
        except (IndexError, TypeError):
            continue
        print(f"{nm:<12}{p:>10.4f}{rc:>10.4f}{a50:>10.4f}{a95:>11.4f}")
    print("-" * 74)
    print(f"{'整体':<12}{r.box.mp:>10.4f}{r.box.mr:>10.4f}{r.box.map50:>10.4f}{r.box.map:>11.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="geom", choices=["geom", "eval"])
    ap.add_argument("--cls", type=int, default=0, help="诊断的类别 id，默认 0=landslide")
    ap.add_argument("--weights", default="runs/train/yolov8s_v2/weights/best.pt")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--only", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    if a.mode == "geom":
        do_geom(a.cls)
    else:
        only = set(x for x in a.only.split(",") if x)
        exclude = set(x for x in a.exclude.split(",") if x)
        do_eval(a.weights, a.split, only, exclude, a.imgsz, a.batch, a.device)


if __name__ == "__main__":
    main()
