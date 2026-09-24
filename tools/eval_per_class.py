"""逐类别 AP 评估 —— 定位 mAP 短板在哪一类

训练的 results.csv 只有整体 mAP，看不出是 rockfall 拖后腿还是 collapse。
本脚本输出每个类别的 P / R / mAP50 / mAP50-95 及验证集实例数，并落一份 JSON 便于多轮对比。

用法:
    # 常规评估（模型类别与 GT 一致）
    python tools/eval_per_class.py --weights runs/train/yolov8s_v1/weights/best.pt \
        --data dataset/splits/split_v3.yaml --split test \
        --split-file dataset/splits/split_v3_test.txt --name v8s1_test

    # v0 四分类模型对齐三分类 GT（类别编号不同，必须做类别映射）
    python tools/eval_per_class.py --weights models/yolov8s_v0/weights/best.pt \
        --data dataset/splits/split_v3.yaml --split test \
        --split-file dataset/splits/split_v3_test.txt --remap 1,2,3 --name v8s0_test

输出:
    控制台表格 + runs/val/<name>/per_class.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def count_instances(label_dir: Path):
    """统计验证集每个类别的标注实例数（单一标签目录时使用）"""
    counts = {}
    for f in label_dir.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            c = line.split()[0]
            if c.isdigit():
                counts[int(c)] = counts.get(int(c), 0) + 1
    return counts


def count_instances_split(split_file: Path):
    """按评估集的图片清单统计实例数。

    split_v3 的图片是从原 train/valid/test 三个划分里重新抽出来的，
    标签仍分散在 dataset/labels/{train,valid,test}，所以必须按图片路径反查标签，
    不能只扫某一个目录（会少算）。
    """
    counts = {}
    missing = 0
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        lbl = Path(str(p).replace("\\", "/").replace("/images/", "/labels/")).with_suffix(".txt")
        if not lbl.exists():
            missing += 1
            continue
        for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            c = ln.split()[0]
            if c.isdigit():
                counts[int(c)] = counts.get(int(c), 0) + 1
    if missing:
        print(f"[warn] {missing} 张图没有对应标签文件")
    return counts


def remap_cls_head(model, keep):
    """把检测头的分类分支按 keep 重排/裁剪类别通道，使模型类别与 GT 对齐。

    背景: v0 模型是 4 类(0=fallen tree, 1=landslide, 2=road collapse, 3=stone)，
    当前 GT 是 3 类(0=landslide, 1=collapse, 2=rockfall)。若不映射直接评估，
    模型的类 1(landslide) 会被当成 GT 类 1(collapse)，mAP 接近 0 且完全无意义。

    做法: 保留 keep 指定的原始类别通道并裁剪到 3 个输出通道，
    同步更新 Detect.nc / Detect.no(no = nc + reg_max*4，不改会导致维度错位)。
    """
    import torch
    import torch.nn as nn

    det = None
    for m in model.model.modules():
        if m.__class__.__name__ == "Detect":
            det = m
            break
    if det is None:
        raise SystemExit("[err] 未找到 Detect 层，无法做类别映射")

    old_nc = int(getattr(det, "nc", 0))
    if not old_nc:
        raise SystemExit("[err] Detect 层没有 nc 属性")
    if max(keep) >= old_nc:
        raise SystemExit(f"[err] --remap {keep} 超出模型类别数 {old_nc}")

    idx = torch.tensor(keep, dtype=torch.long)
    n = 0
    for branch in det.cv3:
        last = None
        for mod in branch.modules():
            if isinstance(mod, nn.Conv2d):
                last = mod
        if last is None or last.out_channels != old_nc:
            continue
        new = nn.Conv2d(
            last.in_channels, len(keep), last.kernel_size, last.stride,
            last.padding, bias=last.bias is not None,
        )
        with torch.no_grad():
            new.weight.copy_(last.weight.index_select(0, idx))
            if last.bias is not None:
                new.bias.copy_(last.bias.index_select(0, idx))
        # 最后一层可能是裸 Conv2d(直接子模块)，也可能是 ultralytics 的 Conv(包在 .conv 里)
        for i in range(len(branch) - 1, -1, -1):
            ch = branch[i]
            if ch is last:
                branch[i] = new
                n += 1
                break
            if hasattr(ch, "conv") and ch.conv is last:
                ch.conv = new
                n += 1
                break

    det.nc = len(keep)
    if hasattr(det, "reg_max"):
        det.no = len(keep) + int(det.reg_max) * 4
    try:
        model.model.nc = len(keep)
    except Exception:
        pass
    # 注意: 不要写 model.model.args["nc"]。ultralytics 会校验 args 的 key，
    # "nc" 不是合法的 YOLO 参数，写进去会让 predict/val 直接抛
    # "SyntaxError: 'nc' is not a valid YOLO argument"。检测头的 nc/no 才是前向真正用到的。
    print(f"[remap] 检测头 {old_nc} -> {len(keep)} 类 keep={keep}，替换 {n} 个分类分支")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="训练占着 GPU 时调小，避免抢显存")
    ap.add_argument("--device", default="0", help="与训练并行时用 cpu，避免抢显存")
    ap.add_argument("--workers", type=int, default=2, help="dataloader worker 数，内存紧张时设 0")
    ap.add_argument("--conf", type=float, default=0.001, help="与训练期验证一致，不要用 0.25")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--label-dir", default="dataset/labels/valid")
    ap.add_argument("--split-file", default="",
                    help="评估集图片路径清单(txt)。给定时用它精确统计各类实例数")
    ap.add_argument("--remap", default="",
                    help="类别映射，如 1,2,3：取模型原类别 1/2/3 作为新类别 0/1/2")
    ap.add_argument("--remap-names", default="landslide,collapse,rockfall",
                    help="映射后的类别名，顺序对应新的 0/1/2")
    ap.add_argument("--project", default="runs/val")
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    from ultralytics import YOLO

    name = a.name or Path(a.weights).parent.parent.name + "_eval"
    model = YOLO(a.weights)

    if a.remap:
        keep = [int(x) for x in a.remap.split(",") if x.strip()]
        model = remap_cls_head(model, keep)
        names_new = [s.strip() for s in a.remap_names.split(",") if s.strip()]
        try:
            model.model.names = {i: nm for i, nm in enumerate(names_new)}
        except Exception:
            pass

    r = model.val(
        data=a.data,
        split=a.split,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        conf=a.conf,
        iou=a.iou,
        plots=True,
        project=a.project,
        name=name,
        exist_ok=True,
        verbose=False,
    )

    names = r.names
    box = r.box
    # ap_class_index 只包含验证集中真实出现过的类别
    idx = list(getattr(box, "ap_class_index", []))
    ap50 = list(getattr(box, "ap50", [])) or list(getattr(box, "ap", []))
    ap95 = list(getattr(box, "ap", []))
    prec = list(getattr(box, "p", []))
    rec = list(getattr(box, "r", []))

    if a.split_file:
        inst = count_instances_split(ROOT / a.split_file)
    else:
        inst = count_instances(ROOT / a.label_dir)

    def get(arr, i, default=0.0):
        try:
            return float(arr[i])
        except (IndexError, TypeError):
            return default

    print("\n" + "=" * 72)
    print(f"weights : {a.weights}" + (f"   remap={a.remap}" if a.remap else ""))
    print(f"split   : {a.split}  imgsz={a.imgsz}  conf={a.conf}  iou={a.iou}")
    print("=" * 72)
    print(f"{'cls':<4}{'name':<12}{'inst':>7}{'P':>9}{'R':>9}{'mAP50':>9}{'mAP50-95':>10}")
    print("-" * 72)
    rows = []
    for i, ci in enumerate(idx):
        nm = names.get(int(ci), str(ci))
        row = {
            "cls": int(ci),
            "name": nm,
            "instances": inst.get(int(ci), 0),
            "precision": round(get(prec, i), 4),
            "recall": round(get(rec, i), 4),
            "map50": round(get(ap50, i), 4),
            "map50_95": round(get(ap95, i), 4),
        }
        rows.append(row)
        print(f"{row['cls']:<4}{nm:<12}{row['instances']:>7}{row['precision']:>9.4f}"
              f"{row['recall']:>9.4f}{row['map50']:>9.4f}{row['map50_95']:>10.4f}")
    print("-" * 72)
    summary = {
        "weights": str(a.weights),
        "remap": a.remap or None,
        "split": a.split,
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "mean_precision": round(float(box.mp), 4),
        "mean_recall": round(float(box.mr), 4),
    }
    print(f"{'ALL':<4}{'':<12}{sum(inst.values()):>7}{box.mp:>9.4f}{box.mr:>9.4f}"
          f"{box.map50:>9.4f}{box.map:>10.4f}")
    print("=" * 72 + "\n")

    out_dir = ROOT / a.project / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "per_class.json"
    out_p.write_text(
        json.dumps({"summary": summary, "per_class": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {out_p}")


if __name__ == "__main__":
    main()
