"""从已有的 results.csv 补画训练曲线图。

用途: 训练中途崩溃(如内存 OOM)时，ultralytics 的收尾阶段不会执行，
results.png / confusion_matrix.png 等图都不会生成，但 results.csv 是逐轮增量写入的，
可以事后用它把曲线补回来，不必重训。

用法:
    python tools/plot_results.py --csv runs/train/yolo11s_v2/results.csv
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="", help="留空则输出到 csv 所在目录")
    ap.add_argument("--name", default="results.png")
    a = ap.parse_args()

    csv_p = Path(a.csv)
    if not csv_p.exists():
        raise SystemExit(f"[err] 找不到 {csv_p}")
    out_dir = Path(a.out_dir) if a.out_dir else csv_p.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics.utils.plotting import plot_results

    res = plot_results(file=str(csv_p), dir=str(out_dir))
    dst = out_dir / a.name
    # plot_results 会写到 dir/results.png；若名字不同则改名
    src = out_dir / "results.png"
    if res is not None and Path(res).exists() and Path(res).resolve() != dst.resolve():
        shutil.move(str(res), str(dst))
    elif src.exists() and a.name != "results.png":
        shutil.move(str(src), str(dst))

    print(f"[plot] -> {dst}" if dst.exists() else "[warn] 未生成 results.png")


if __name__ == "__main__":
    main()
