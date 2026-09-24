"""分析训练结果曲线，判断收敛状态与过拟合程度

用法:
    python tools/analyze_results.py                       # 自动分析 runs/train 下最新的 results.csv
    python tools/analyze_results.py runs/train/xxx/results.csv
"""
import argparse
import csv
from pathlib import Path


def g(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None)
    a = ap.parse_args()

    if a.csv:
        p = Path(a.csv)
    else:
        cands = sorted(Path("runs/train").glob("*/results.csv"), key=lambda x: x.stat().st_mtime)
        if not cands:
            raise SystemExit("未找到 runs/train/*/results.csv")
        p = cands[-1]

    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    print(f"file  : {p}")
    print(f"epochs: {len(rows)}")

    best = max(rows, key=lambda r: g(r, "metrics/mAP50(B)") or 0)
    print(
        f"BEST  : mAP50={g(best, 'metrics/mAP50(B)'):.4f}  "
        f"mAP50-95={g(best, 'metrics/mAP50-95(B)'):.4f}  @ epoch {best['epoch']}"
    )

    last = rows[-1]
    print(
        f"LAST  : mAP50={g(last, 'metrics/mAP50(B)'):.4f}  "
        f"mAP50-95={g(last, 'metrics/mAP50-95(B)'):.4f}  @ epoch {last['epoch']}"
    )

    # 过拟合判断: 后 1/4 的 val loss 是否相对最低点回升
    n = len(rows)
    tail = rows[int(n * 0.75):]
    if g(tail[0], "val/box_loss") is not None:
        vmin = min(g(r, "val/box_loss") for r in rows if g(r, "val/box_loss") is not None)
        vmax_tail = max(g(r, "val/box_loss") for r in tail if g(r, "val/box_loss") is not None)
        print(f"\nval/box_loss: 全程最低={vmin:.4f}  后25%最高={vmax_tail:.4f}")
        if vmax_tail > vmin * 1.15:
            print("  -> val loss 明显回升，存在过拟合，继续加 epoch 无益")
        else:
            print("  -> val loss 未见明显回升，未过拟合")

    print("\nepoch | mAP50   mAP50-95 |   P      R    | box    cls    dfl  | vbox  vcls  vdfl")
    step = max(1, n // 15)
    for r in rows[::step]:
        print(
            f"{r['epoch']:>5} | {g(r,'metrics/mAP50(B)'):.4f}  {g(r,'metrics/mAP50-95(B)'):.4f}   "
            f"| {g(r,'metrics/precision(B)'):.3f}  {g(r,'metrics/recall(B)'):.3f} "
            f"| {g(r,'train/box_loss'):.3f}  {g(r,'train/cls_loss'):.3f}  {g(r,'train/dfl_loss'):.3f} "
            f"| {g(r,'val/box_loss'):.3f}  {g(r,'val/cls_loss'):.3f}  {g(r,'val/dfl_loss'):.3f}"
        )
    print(
        f"{last['epoch']:>5} | {g(last,'metrics/mAP50(B)'):.4f}  {g(last,'metrics/mAP50-95(B)'):.4f}   "
        f"| {g(last,'metrics/precision(B)'):.3f}  {g(last,'metrics/recall(B)'):.3f} "
        f"| {g(last,'train/box_loss'):.3f}  {g(last,'train/cls_loss'):.3f}  {g(last,'train/dfl_loss'):.3f} "
        f"| {g(last,'val/box_loss'):.3f}  {g(last,'val/cls_loss'):.3f}  {g(last,'val/dfl_loss'):.3f}"
    )


if __name__ == "__main__":
    main()
