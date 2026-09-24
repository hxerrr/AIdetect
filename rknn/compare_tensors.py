"""对比 ONNX 与 RKNN(simulator / 板端 NPU) 的输出张量，量化转换的精度损失

用法:
    # 1. 先用 onnx2rknn.py --sim-infer --dump 分别导出 fp16 / int8 的输出
    python rknn/onnx2rknn.py --onnx weights/yolo11s.onnx --output weights/yolo11s_fp16.rknn \
        --sim-infer rknn/calib/calib_00000.jpg --dump rknn/dump_fp16.npy
    python rknn/onnx2rknn.py --onnx weights/yolo11s.onnx --output weights/yolo11s_i8.rknn \
        --quant --dataset rknn/calib.txt --sim-infer rknn/calib/calib_00000.jpg --dump rknn/dump_i8.npy

    # 2. 用 ONNXRuntime 在同一张图上跑出基准张量
    python rknn/compare_tensors.py --onnx weights/yolo11s.onnx --image rknn/calib/calib_00000.jpg \
        --dump rknn/dump_onnx.npy

    # 3. 对比
    python rknn/compare_tensors.py --a rknn/dump_onnx.npy --b rknn/dump_fp16.npy --decode
    python rknn/compare_tensors.py --a rknn/dump_onnx.npy --b rknn/dump_i8.npy --decode

判读标准(经验):
    cosine >= 0.999 且 box 分支平均绝对误差 < 0.5 像素  -> 转换无损，可放心上板
    cosine 0.99~0.999                                  -> 轻微掉点，需跑数据集级对比确认
    cosine < 0.99 或 cls 分支误差 > 0.05               -> 掉点风险大，换 mmse 量化/增大标定集/改混合精度
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from rknn_postprocess import decode_yolo, letterbox


def stats(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = np.abs(a - b)
    cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12))
    return {
        "shape_a": tuple(a.shape),
        "shape_b": tuple(b.shape),
        "absmax_a": float(np.abs(a).max()),
        "absmax_b": float(np.abs(b).max()),
        "cosine": cos,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rel_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)),
    }


def branch_report(tag, a, b):
    d = stats(a, b)
    print(f"[{tag}] shape={d['shape_a']}->{d['shape_b']} cosine={d['cosine']:.6f} "
          f"max_diff={d['max_abs_diff']:.4f} mean_diff={d['mean_abs_diff']:.5f} rel_l2={d['rel_l2']:.4f}")
    return d


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-6)


def compare_dets(a, b, conf=0.25, iou_nms=0.45, iou_thr=0.5):
    da = decode_yolo(a, conf_thresh=conf, iou_thresh=iou_nms)
    db = decode_yolo(b, conf_thresh=conf, iou_thresh=iou_nms)
    matched = 0
    conf_errs = []
    for x in da:
        for y in db:
            if x[0] == y[0] and iou_xyxy(x[2:], y[2:]) >= iou_thr:
                matched += 1
                conf_errs.append(abs(x[1] - y[1]))
                break
    rec = matched / len(da) if da else 1.0
    prec = matched / len(db) if db else 1.0
    print(f"[dets] onnx={len(da)} rknn={len(db)} matched={matched} recall={rec:.3f} precision={prec:.3f} "
          f"avg_conf_err={np.mean(conf_errs) if conf_errs else 0:.4f}")
    return len(da), len(db), matched, rec, prec


def dump_onnx(onnx_path, image, imgsz, out):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    img0 = cv2.imread(image)
    if img0 is None:
        raise ValueError(f"无法读图: {image}")
    lb, _, _, _ = letterbox(img0, target_size=imgsz)
    blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    y = sess.run(None, {name: blob})[0]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, y)
    print(f"[dump] onnx output {y.shape} -> {out}")
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="基准张量 npy(通常来自 ONNXRuntime)")
    ap.add_argument("--b", help="待比对张量 npy(来自 RKNN simulator 或板端)")
    ap.add_argument("--decode", action="store_true", help="同时比较解码后的检测框")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--onnx", default="", help="提供则先跑 ONNXRuntime 生成基准张量")
    ap.add_argument("--image", default="rknn/calib/calib_00000.jpg")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dump", default="", help="--onnx 模式下基准张量保存路径")
    a = ap.parse_args()

    if a.onnx:
        ya = dump_onnx(a.onnx, a.image, a.imgsz, a.dump or "rknn/dump_onnx.npy")
    elif a.a:
        ya = np.load(a.a)
    else:
        ap.error("需要 --a 或 --onnx")

    if not a.b:
        print("[info] 缺少 --b，仅输出基准张量信息")
        print(f"       shape={ya.shape} absmax={np.abs(ya).max():.4f}")
        return

    yb = np.load(a.b)
    if ya.shape != yb.shape:
        print(f"[warn] shape 不一致: {ya.shape} vs {yb.shape}，尝试对齐后比较")
        n = min(ya.size, yb.size)
        ya = ya.reshape(-1)[:n]
        yb = yb.reshape(-1)[:n]

    branch_report("all", ya, yb)
    flat_a = ya.reshape(ya.shape[0], -1) if ya.ndim == 3 else ya[None]
    flat_b = yb.reshape(yb.shape[0], -1) if yb.ndim == 3 else yb[None]
    if flat_a.shape[0] >= 7:
        branch_report("box", flat_a[:4], flat_b[:4])
        branch_report("cls", flat_a[4:], flat_b[4:])

    if a.decode:
        compare_dets(ya, yb, conf=a.conf, iou_nms=a.iou)


if __name__ == "__main__":
    main()
