"""检查 .pt 检查点里各字段实际占多少字节，判断"为什么这么大"

.pt 里可能同时装了:
  model      模型权重（fp32 ~37.8MB / fp16 ~18.9MB，yolo11s 量级）
  ema        EMA 滑动平均权重（推理时用不到）
  optimizer  AdamW 状态（每个参数 2 份 fp32，约模型的 2 倍，仅续训需要）

注意: optimizer 的 state 是嵌套 dict，只数顶层张量会漏掉，这里用 pickle 实测真实字节。

用法:
    python tools/inspect_ckpt.py runs/train/xxx/weights/best.pt [more.pt ...]
"""
import pickle
import sys
from pathlib import Path

import torch


def main():
    paths = sys.argv[1:] or ["runs/train/yolo11s_e150b/weights/best.pt"]
    for p in paths:
        f = Path(p)
        print("=" * 74)
        print(p)
        if not f.exists():
            print("   [不存在]")
            continue
        mb = f.stat().st_size / 1e6
        try:
            ck = torch.load(str(f), map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"   [加载失败] {str(e)[:150]}")
            continue
        print(f"   文件大小 {mb:.1f} MB   字段: {list(ck.keys())}")
        rows = []
        for k, v in ck.items():
            try:
                b = len(pickle.dumps(v, protocol=4))
            except Exception:  # noqa: BLE001
                b = -1
            rows.append((k, b))
        rows.sort(key=lambda x: -x[1])
        for k, b in rows:
            if b <= 0:
                continue
            flag = "  <== 大头" if b > 5e6 else ""
            print(f"     {k:<14} {b/1e6:>8.1f} MB{flag}")
        print(f"     {'合计':<14} {sum(b for _, b in rows if b > 0)/1e6:>8.1f} MB")
        print("   （model/ema 是 nn.Module，pickle 后含结构信息，略大于纯权重）")
        print("   推理/导出只需要 model（或 ema）；optimizer 仅 resume 需要")


if __name__ == "__main__":
    main()
