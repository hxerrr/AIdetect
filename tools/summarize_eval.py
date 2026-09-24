"""汇总多个评估结果的 per_class.json，打成对比表

用法:
    python tools/summarize_eval.py --names yolov8s-v1_test,yolov11s-v1_test,yolov8s-v0_test,yolov11s-v0_test
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["landslide", "collapse", "rockfall"]


def load(name):
    p = ROOT / "runs" / "val" / name / "per_class.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "name": name,
        "summary": d["summary"],
        "per_class": {r["name"]: r for r in d["per_class"]},
    }


def fmt(v, w=9):
    return f"{v:>{w}.4f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="runs/val 下的评估目录名，逗号分隔")
    ap.add_argument("--project", default="runs/val")
    a = ap.parse_args()

    rows = []
    for n in [x.strip() for x in a.names.split(",") if x.strip()]:
        r = load(n)
        if r is None:
            print(f"[warn] 缺少结果: {n}")
            continue
        rows.append(r)

    if not rows:
        raise SystemExit("[err] 没有任何结果")

    print()
    print("=== 整体指标 ===")
    print(f"{'模型':<16}{'P':>9}{'R':>9}{'mAP50':>9}{'mAP50-95':>10}")
    print("-" * 53)
    for r in rows:
        s = r["summary"]
        print(f"{r['name']:<16}{fmt(s['mean_precision'])}{fmt(s['mean_recall'])}"
              f"{fmt(s['map50'])}{fmt(s['map50_95'], 10)}")

    for key, title in (("map50", "mAP50"), ("map50_95", "mAP50-95")):
        print()
        print(f"=== 分类别 {title} ===")
        print(f"{'模型':<16}" + "".join(f"{c:>12}" for c in CLASSES))
        print("-" * (16 + 12 * len(CLASSES)))
        for r in rows:
            vals = [r["per_class"].get(c, {}).get(key) for c in CLASSES]
            print(f"{r['name']:<16}" + "".join(fmt(v, 12) for v in vals))

    print()
    print("=== 各类实例数(评估集 GT) ===")
    print(f"{'模型':<16}" + "".join(f"{c:>12}" for c in CLASSES))
    print("-" * (16 + 12 * len(CLASSES)))
    for r in rows:
        vals = [r["per_class"].get(c, {}).get("instances") for c in CLASSES]
        print(f"{r['name']:<16}" + "".join(
            f"{(v if v is not None else '--'):>12}" for v in vals))
    print()


if __name__ == "__main__":
    main()
