"""ONNX -> RKNN 转换（在虚拟机 Ubuntu + rknn-toolkit2 1.5.2 上运行）

推荐的两步走流程（先确认算子能编译，再谈量化精度）:

    第一步: fp16 不量化，验证编译器能否吃下 YOLO11 的全部算子
    python rknn/onnx2rknn.py --onnx weights/yolo11s_aidetect.onnx --output weights/yolo11s_fp16.rknn

    第二步: int8 量化（用 valid 集标定）
    python rknn/onnx2rknn.py --onnx weights/yolo11s_aidetect.onnx --output weights/yolo11s_i8.rknn \
        --quant --dataset rknn/calib.txt

    可选: 转换完用 PC simulator 跑一张图，dump 输出张量用于和 ONNX 逐元素比对
    python rknn/onnx2rknn.py --onnx ... --output ... --sim-infer rknn/calib/calib_00000.jpg --dump out_rknn.npy

    若虚拟机已 adb 连接板子，可直接在板端 NPU 上跑:
    python rknn/onnx2rknn.py --onnx ... --output ... --device-id <serial> --sim-infer xxx.jpg --dump out_board.npy

要点:
  - build 后务必看日志里的 "W:" 警告，出现算子不支持会 fallback 到 CPU，是掉点和掉帧的主因
  - int8 掉点优先靠 --quant-algo mmse 与提高标定集数量解决，仍不行再考虑混合精度/升级 toolkit2 2.3.2
"""
import argparse
import sys
from pathlib import Path


def do_config(rknn, a):
    """不同 toolkit2 小版本的 config 参数集合有差异，先带全参数，失败则回退最小集"""
    kw = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": a.target,
    }
    if a.optimization_level is not None:
        kw["optimization_level"] = a.optimization_level
    if a.quant:
        kw["quantized_dtype"] = a.quant_dtype
        kw["quantized_algorithm"] = a.quant_algo

    try:
        rknn.config(**kw)
        print(f"[config] {kw}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 扩展参数不被当前 toolkit 支持，回退最小参数集: {e}")
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=a.target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--quant", action="store_true", help="int8 量化")
    ap.add_argument("--dataset", default="rknn/calib.txt", help="量化标定集列表")
    ap.add_argument("--quant-dtype", default="asymmetric_quantized-8")
    ap.add_argument("--quant-algo", default="normal", choices=["normal", "kl_divergence", "mmse"])
    ap.add_argument("--optimization-level", type=int, default=None)
    ap.add_argument("--sim-infer", default="", help="转换后推理的图片路径")
    ap.add_argument("--dump", default="", help="推理输出保存为 npy")
    ap.add_argument("--device-id", default="", help="adb 设备号，留空则用 PC simulator")
    a = ap.parse_args()

    onnx_p = Path(a.onnx)
    if not onnx_p.exists():
        sys.exit(f"[err] ONNX 不存在: {onnx_p}")

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        do_config(rknn, a)

        print(f"[load] {onnx_p}")
        ret = rknn.load_onnx(model=str(onnx_p))
        if ret != 0:
            sys.exit(f"[err] load_onnx 失败, ret={ret}")

        if a.quant:
            ds = Path(a.dataset)
            if not ds.exists():
                sys.exit(f"[err] 量化标定列表不存在: {ds}（先跑 tools/export_onnx.py 生成）")
            n = sum(1 for _ in ds.open(encoding="utf-8") if _.strip())
            print(f"[build] int8 量化, 标定集 {n} 张")
        else:
            print("[build] fp16（不量化）")

        ret = rknn.build(do_quantization=a.quant, dataset=str(a.dataset) if a.quant else None)
        if ret != 0:
            sys.exit(f"[err] build 失败, ret={ret}")

        out_p = Path(a.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(out_p))
        if ret != 0:
            sys.exit(f"[err] export_rknn 失败, ret={ret}")
        print(f"[export] {out_p} ({out_p.stat().st_size/1e6:.2f} MB)")

        if a.sim_infer:
            import cv2
            import numpy as np

            if a.device_id:
                print(f"[runtime] 板端 NPU, device={a.device_id}")
                ret = rknn.init_runtime(target=a.target, device_id=a.device_id)
            else:
                print("[runtime] PC simulator")
                ret = rknn.init_runtime()
            if ret != 0:
                sys.exit(f"[err] init_runtime 失败, ret={ret}")

            img = cv2.imread(a.sim_infer)
            if img is None:
                sys.exit(f"[err] 读图失败: {a.sim_infer}")
            img = cv2.resize(img, (a.imgsz, a.imgsz))

            outs = rknn.inference(inputs=[img])
            for i, o in enumerate(outs):
                print(f"[sim] out{i} shape={o.shape} dtype={o.dtype} absmax={abs(o).max():.4f}")
            if a.dump:
                np.save(a.dump, outs[0])
                print(f"[sim] 已保存 {a.dump}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
