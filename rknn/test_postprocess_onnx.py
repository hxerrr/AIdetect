"""验证 ONNX 后处理精度：与 Ultralytics YOLO 输出对齐

用法:
    python rknn/test_postprocess_onnx.py --onnx weights/yolo11s_aidetect_base.onnx --image dataset/images/test/xxx.jpg

步骤:
    1. 读图 -> letterbox -> ONNXRuntime 推理。
    2. 用 rknn_postprocess 解码框。
    3. 同一幅图用 Ultralytics 预测。
    4. 比较两路框的 IoU 与置信度，确认后处理逻辑正确。
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from rknn_postprocess import decode_yolo, letterbox, scale_coords

NAMES = {0: "landslide", 1: "collapse", 2: "rockfall"}


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-6)


def match_and_compare(dets_onnx, dets_yolo, iou_thr=0.5, conf_thresh=0.25, margin=0.05):
    """比较两路检测结果。

    margin: 置信度落在 [conf_thresh, conf_thresh+margin] 的未匹配框视为阈值边界抖动（fp32 数值差异导致），
            单独统计，不计入真实漏检/误检，避免把 0.249 vs 0.251 的抖动误判成后处理错误。
    """
    matched = 0
    conf_errs = []
    matched_onnx = set()

    for oi, oo in enumerate(dets_onnx):
        obox = [oo[2], oo[3], oo[4], oo[5]]
        for yo in dets_yolo:
            ybox = [yo[2], yo[3], yo[4], yo[5]]
            if iou(ybox, obox) >= iou_thr and yo[0] == oo[0]:
                matched += 1
                matched_onnx.add(oi)
                conf_errs.append(abs(yo[1] - oo[1]))
                break

    miss_real = miss_border = 0
    for yo in dets_yolo:
        ybox = [yo[2], yo[3], yo[4], yo[5]]
        hit = any(iou(ybox, [oo[2], oo[3], oo[4], oo[5]]) >= iou_thr and yo[0] == oo[0] for oo in dets_onnx)
        if not hit:
            if yo[1] < conf_thresh + margin:
                miss_border += 1
            else:
                miss_real += 1

    extra_real = extra_border = 0
    for oi, oo in enumerate(dets_onnx):
        if oi in matched_onnx:
            continue
        if oo[1] < conf_thresh + margin:
            extra_border += 1
        else:
            extra_real += 1

    n_yolo = len(dets_yolo)
    n_onnx = len(dets_onnx)
    recall = matched / n_yolo if n_yolo else 1.0
    precision = matched / n_onnx if n_onnx else 1.0
    # 剔除边界抖动后的“硬指标”
    recall_adj = matched / (matched + miss_real) if (matched + miss_real) else 1.0
    precision_adj = matched / (matched + extra_real) if (matched + extra_real) else 1.0
    avg_conf_err = float(np.mean(conf_errs)) if conf_errs else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "recall_adj": recall_adj,
        "precision_adj": precision_adj,
        "conf_err": avg_conf_err,
        "matched": matched,
        "n_yolo": n_yolo,
        "n_onnx": n_onnx,
        "miss_real": miss_real,
        "miss_border": miss_border,
        "extra_real": extra_real,
        "extra_border": extra_border,
    }


def run(args, model=None):
    img0 = cv2.imread(args.image)
    if img0 is None:
        raise ValueError(f"无法读图: {args.image}")
    h0, w0 = img0.shape[:2]

    # ---- ONNX 推理 ----
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    img_lb, scale, pad_left, pad_top = letterbox(img0, target_size=args.imgsz)
    blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    t0 = time.time()
    outs = sess.run(None, {inp_name: blob})
    onnx_infer = time.time() - t0
    dets_raw = decode_yolo(outs[0], conf_thresh=args.conf, iou_thresh=args.iou, max_det=args.max_det)
    # 将 ONNX 输入尺寸下的 xyxy 映射回原图
    dets_onnx = []
    for cls_id, conf, x1, y1, x2, y2 in dets_raw:
        arr = scale_coords([[x1, y1, x2, y2]], (h0, w0), input_size=args.imgsz, pad_left=pad_left, pad_top=pad_top, scale=scale)
        nx1, ny1, nx2, ny2 = arr[0]
        dets_onnx.append((cls_id, conf, float(nx1), float(ny1), float(nx2), float(ny2)))

    # ---- Ultralytics 推理 ----
    if model is None:
        model = YOLO(args.weights)
    t0 = time.time()
    res = model.predict(img0, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
    yolo_infer = time.time() - t0
    dets_yolo = []
    for box, cls_id, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        dets_yolo.append((int(cls_id), float(conf), float(box[0]), float(box[1]), float(box[2]), float(box[3])))

    m = match_and_compare(dets_onnx, dets_yolo, conf_thresh=args.conf)

    print(f"image: {args.image} ({w0}x{h0})")
    print(f"ONNX  infer={onnx_infer*1000:.1f}ms  dets={m['n_onnx']}")
    print(f"YOLO  infer={yolo_infer*1000:.1f}ms  dets={m['n_yolo']}")
    print(f"match={m['matched']}  recall={m['recall']:.3f}  precision={m['precision']:.3f}  "
          f"avg_conf_err={m['conf_err']:.4f}  miss(real/border)={m['miss_real']}/{m['miss_border']}  "
          f"extra(real/border)={m['extra_real']}/{m['extra_border']}")
    print("--- ONNX ---")
    for d in dets_onnx[:10]:
        print(f"  {NAMES[d[0]]:10} conf={d[1]:.3f} box=[{d[2]:.1f},{d[3]:.1f},{d[4]:.1f},{d[5]:.1f}]")
    if args.verbose:
        print("--- ONNX ---")
        for d in dets_onnx[:10]:
            print(f"  {NAMES[d[0]]:10} conf={d[1]:.3f} box=[{d[2]:.1f},{d[3]:.1f},{d[4]:.1f},{d[5]:.1f}]")
        print("--- YOLO ---")
        for d in dets_yolo[:10]:
            print(f"  {NAMES[d[0]]:10} conf={d[1]:.3f} box=[{d[2]:.1f},{d[3]:.1f},{d[4]:.1f},{d[5]:.1f}]")
    return m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="weights/yolo11s_aidetect_base.onnx")
    p.add_argument("--weights", default="runs/train/yolo11s_e150b/weights/best.pt")
    p.add_argument("--image", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--limit", type=int, default=0, help="--image 为目录时最多处理多少张，0=全部")
    p.add_argument("--verbose", action="store_true", help="打印每张图的框详情")
    return p.parse_args()


def main():
    args = parse_args()
    p = Path(args.image)
    if not p.is_dir():
        run(args)
        return

    files = [q for q in sorted(p.rglob("*")) if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"目录内无图片: {p}")

    keys = ["recall", "precision", "recall_adj", "precision_adj", "conf_err"]
    agg = {k: [] for k in keys}
    tot = {"matched": 0, "n_yolo": 0, "n_onnx": 0, "miss_real": 0, "miss_border": 0, "extra_real": 0, "extra_border": 0}
    model = YOLO(args.weights)
    worst = []
    for f in files:
        args.image = str(f)
        r = run(args, model)
        for k in keys:
            agg[k].append(r[k])
        for k in tot:
            tot[k] += r[k]
        worst.append((r["recall_adj"], str(f)))
    worst.sort()

    n = len(files)
    print(f"\n===== 汇总 {n} 张图 =====")
    print(f"mean recall(adj)    = {np.mean(agg['recall_adj']):.4f}   (原始 {np.mean(agg['recall']):.4f})")
    print(f"mean precision(adj) = {np.mean(agg['precision_adj']):.4f}   (原始 {np.mean(agg['precision']):.4f})")
    print(f"mean conf_err       = {np.mean(agg['conf_err']):.4f}")
    print(f"matched={tot['matched']}  yolo_dets={tot['n_yolo']}  onnx_dets={tot['n_onnx']}")
    print(f"漏检 real/border = {tot['miss_real']}/{tot['miss_border']}   "
          f"多检 real/border = {tot['extra_real']}/{tot['extra_border']}")
    print("最差 3 张: " + ", ".join(f"{Path(p).name}:{v:.2f}" for v, p in worst[:3]))

    miss_rate = tot["miss_real"] / max(1, tot["n_yolo"])
    extra_rate = tot["extra_real"] / max(1, tot["n_onnx"])
    print(f"真实漏检率={miss_rate:.3%}  真实多检率={extra_rate:.3%}")
    ok = miss_rate <= 0.02 and extra_rate <= 0.02 and np.mean(agg["conf_err"]) <= 0.05
    print(f"结论: {'后处理与 PyTorch 一致，可进入 RKNN 转换' if ok else '存在真实偏差，检查 letterbox/NMS/置信度阈值'}")


if __name__ == "__main__":
    main()
