"""导出 RKNN 友好的 ONNX，并生成 int8 量化标定集。

用法:
    python tools/export_onnx.py --weights runs/train/yolo11s_e150b/weights/best.pt
    python tools/export_onnx.py --weights ... --opset 12 --calib-num 200

说明:
  - nms=False: 输出检测头原始张量，NMS/后处理放到 CPU 做(RKNN 上跑 NMS 算子支持差)
  - dynamic=False: 固定输入尺寸，RKNN 要求静态 shape
  - simplify=True: 消除冗余算子，显著降低 RKNN 编译失败概率
  - int8 量化标定集取自 valid 集(与训练解耦，避免用训练图做量化标定)
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def make_calib(src_dir: Path, calib_dir: Path, calib_num: int, seed: int = 0):
    """从 valid 集抽样作为 int8 量化标定集，生成 RKNN 需要的 dataset.txt"""
    if calib_dir.exists():
        shutil.rmtree(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    random.seed(seed)
    random.shuffle(files)
    picked = files[:calib_num]

    lines = []
    for i, p in enumerate(picked):
        dst = calib_dir / f"calib_{i:05d}{p.suffix.lower()}"
        shutil.copy2(p, dst)
        lines.append(str(dst.resolve()).replace("\\", "/"))

    txt = calib_dir.parent / "calib.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"[calib] {len(lines)} 张 -> {calib_dir}, 列表 -> {txt}")
    return txt


def export(weights: str, imgsz: int, opset: int, out: str):
    from ultralytics import YOLO

    model = YOLO(weights)
    path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        dynamic=False,
        nms=False,
        half=False,
    )
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if Path(path).resolve() != out_p.resolve():
        shutil.copy2(path, out_p)
    print(f"[onnx] {out_p} ({out_p.stat().st_size/1e6:.2f} MB)")
    return out_p


def check_onnx(onnx_path: Path, imgsz: int):
    """用 onnxruntime 跑一次推理，确认图合法、输出 shape 符合预期"""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    print("[check] inputs:")
    for i in ins:
        print(f"        {i.name}  {i.shape}  {i.type}")
    print("[check] outputs:")
    for o in outs:
        print(f"        {o.name}  {o.shape}  {o.type}")

    x = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    y = sess.run([o.name for o in outs], {ins[0].name: x})
    print(f"[check] dummy forward OK, out0 shape={y[0].shape}, absmax={np.abs(y[0]).max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/train/yolo11s_e150b/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", default="weights/yolo11s_aidetect.onnx")
    ap.add_argument("--calib-src", default="dataset/images/valid")
    ap.add_argument("--calib-dir", default="rknn/calib")
    ap.add_argument("--calib-num", type=int, default=200)
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()

    if not a.skip_calib:
        make_calib(Path(a.calib_src), Path(a.calib_dir), a.calib_num)

    onnx_p = export(a.weights, a.imgsz, a.opset, a.out)
    check_onnx(onnx_p, a.imgsz)


if __name__ == "__main__":
    main()
