"""用训练好的权重跑 video/input 下的视频和图片，结果写到 video/output/<name>/

用法:
    python tools/predict_v8s_v2.py --weights runs/train/yolov8s_v2/weights/best.pt --name yolov8s-v2
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/train/yolov8s_v2/weights/best.pt")
    ap.add_argument("--name", default="yolov8s-v2", help="video/output 下的输出目录名")
    ap.add_argument("--img-subdir", default="images",
                    help="图片结果放在 <out>/<name>/ 下的哪个子目录，"
                         "跑 dataset/images/test 时用 test_images(和旧模型一致)")
    ap.add_argument("--src", default="video/input")
    ap.add_argument("--out", default="video/output")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_absolute():
        src = ROOT / src

    vids = sorted(p for p in src.glob("*") if p.suffix.lower() in VID_EXT)
    # 源目录下若有 images/ 子目录就用它(视频+图片混放的情况，如 video/input)；
    # 没有就把源目录本身当图片目录(如 dataset/images/test)
    img_src = src / "images" if (src / "images").is_dir() else src
    imgs = sorted(p for p in img_src.glob("*")
                  if p.suffix.lower() in IMG_EXT and not p.name.endswith("_det.jpg")) \
        if img_src.is_dir() else []

    print(f"[in] 视频 {len(vids)} 个, 图片 {len(imgs)} 张", flush=True)
    if not vids and not imgs:
        sys.exit("[err] 没有可推理的文件")

    from ultralytics import YOLO
    m = YOLO(a.weights)
    kw = dict(imgsz=a.imgsz, conf=a.conf, device=a.device,
              save=True, save_txt=True, save_conf=True, exist_ok=True)

    # 两个坑，都是 ultralytics 新版行为:
    # 1) source 别传 list of path —— 列表里的 mp4 会被当图片用 PIL 打开而报错，
    #    传目录字符串才走视频分支(目录只扫一层，images/ 子目录不会被误当视频)
    # 2) project 必须给绝对路径 —— 传相对路径会被拼到 runs/detect/ 下面，
    #    结果落到 runs/detect/video/output/... 而不是 video/output/
    out_root = (ROOT / a.out).resolve()
    t0 = time.time()
    if vids:
        print("[run] 视频推理 ...", flush=True)
        m.predict(source=str(src), project=str(out_root), name=a.name, **kw)
    if imgs:
        print("[run] 图片推理 ...", flush=True)
        m.predict(source=str(img_src), project=str(out_root / a.name),
                  name=a.img_subdir, **kw)
        # ultralytics 存的是原名，而 make_compare.py 按 <原名>_det.jpg 找结果，
        # 这里统一改名，否则后面拼接时会被当成"缺结果"跳过
        d = out_root / a.name / a.img_subdir
        renamed = 0
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in IMG_EXT and not p.name.endswith("_det.jpg"):
                p.rename(p.with_name(f"{p.stem}_det.jpg"))
                renamed += 1
        print(f"[rename] {renamed} 张 -> <原名>_det.jpg", flush=True)

    print(f"[done] 用时 {time.time() - t0:.0f}s -> {Path(a.out) / a.name}", flush=True)


if __name__ == "__main__":
    main()
