"""给 .pt 检查点瘦身：去掉 optimizer / scaler，只保留推理所需权重

best.pt 常有 70+ MB，其中约 57 MB 是 AdamW 优化器状态（每个参数 2 份 fp32），
只有 --resume 断点续训才用得上；推理和导出 ONNX 只需要 model / ema。

用法:
    # 另存为 xxx_slim.pt（默认，不动原文件）
    python tools/slim_ckpt.py runs/train/xxx/weights/best.pt

    # 就地覆盖（会丢掉续训能力，先确认不再 resume）
    python tools/slim_ckpt.py runs/train/xxx/weights/best.pt --inplace
"""
import argparse
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+", help="一个或多个 .pt 路径")
    ap.add_argument("--inplace", action="store_true", help="覆盖原文件（默认另存为 *_slim.pt）")
    ap.add_argument("--half", action="store_true", default=True, help="权重转 fp16（默认开启，体积减半）")
    a = ap.parse_args()

    for p in a.ckpt:
        f = Path(p)
        if not f.exists():
            print(f"[skip] 不存在: {f}")
            continue
        before = f.stat().st_size / 1e6
        ck = torch.load(str(f), map_location="cpu", weights_only=False)

        # 保留推理/导出真正需要的：模型权重（优先 model，没有则用 ema）
        model = ck.get("model")
        if model is None or (hasattr(model, "state_dict") and not any(True for _ in model.state_dict())):
            model = ck.get("ema")
        if model is None:
            print(f"[skip] {f} 里既没有 model 也没有 ema")
            continue

        if a.half and hasattr(model, "half"):
            model = model.half()

        out = {
            "epoch": ck.get("epoch"),
            "best_fitness": ck.get("best_fitness"),
            "model": model,
            "ema": None,
            "updates": ck.get("updates"),
            "optimizer": None,   # 只有 resume 需要，去掉
            "scaler": None,
            "train_args": ck.get("train_args"),
            "train_metrics": ck.get("train_metrics"),
            "train_results": ck.get("train_results"),
            "date": ck.get("date"),
            "version": ck.get("version"),
            "git": ck.get("git"),
            "license": ck.get("license"),
            "docs": ck.get("docs"),
        }
        dst = f if a.inplace else f.with_name(f.stem + "_slim.pt")
        torch.save(out, str(dst))
        after = dst.stat().st_size / 1e6
        print(f"[ok] {f.name}: {before:.1f} MB -> {after:.1f} MB  ({dst})")


if __name__ == "__main__":
    sys.exit(main())
