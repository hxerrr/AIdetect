"""轻量运动目标检测与时序滤波验证工具（本地 PC / 视频）

跟踪与运动判定逻辑统一放在 tools/tracker_common.py，本脚本只负责
"读视频 -> YOLO 推理 -> 喂给 tracker -> 可视化/落盘"，与板端 board_infer.py 保持同一套逻辑。

用法:
    # 本地视频验证
    python tools/motion_tracker.py --weights runs/train/yolo11s_v2/weights/best.pt --source path/to/video.mp4 --view-img

    # 用测试集图片序列模拟
    python tools/motion_tracker.py --weights runs/train/yolo11s_v2/weights/best.pt --source dataset/images/test --output runs/motion/test.mp4

参数说明:
    --motion-threshold  归一化位移/秒（位移 ÷ 图像对角线 ÷ 秒），默认 0.10 ≈ 1080p 下 220 px/s
    --min-motion-frames 连续多少帧超阈值才判为运动，默认 3（滞回，抑制检测抖动误报）
    --no-compensate     关闭相机全局运动补偿（默认开启）
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracker_common import MotionTracker, draw_tracks  # noqa: E402


def load_source(source):
    source = Path(source)
    if source.is_dir():
        files = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png")) + sorted(source.glob("*.jpeg"))
        if not files:
            raise ValueError(f"目录中没有图片: {source}")
        return "images", files
    if source.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {source}")
        return "video", cap
    raise ValueError(f"不支持的输入源: {source}")


def run(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    tracker = MotionTracker(
        iou_threshold=args.iou_threshold,
        max_miss=args.max_miss,
        min_hits=args.min_hits,
        motion_threshold=args.motion_threshold,
        event_cooldown=args.event_cooldown,
        min_motion_frames=args.min_motion_frames,
        default_fps=args.fps,
        compensate=not args.no_compensate,
        # 不同代模型的类别编号不同，必须用模型自带的 names
        names=getattr(model, "names", None),
    )

    kind, src = load_source(args.source)

    # 输出形态: 视频源 -> 视频; 图片序列 -> 若 --output 以视频后缀结尾则出视频，否则按目录存标注图
    VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}
    out_is_video = bool(args.output) and (kind != "images" or Path(args.output).suffix.lower() in VIDEO_EXTS)
    out_dir = None
    if args.output and not out_is_video:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    out_h = out_w = None
    frame_idx = 0
    total_infer = 0.0
    counts = {"new": 0, "moving": 0}

    while True:
        if args.max_frames and frame_idx >= args.max_frames:
            break
        if kind == "video":
            ret, frame = src.read()
            if not ret:
                break
        else:
            if frame_idx >= len(src):
                break
            frame = cv2.imread(str(src[frame_idx]))
            if frame is None:
                frame_idx += 1
                continue

        h, w = frame.shape[:2]
        if writer is not None and (h, w) != (out_h, out_w):
            frame = cv2.resize(frame, (out_w, out_h))
            h, w = out_h, out_w

        t0 = time.time()
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou_nms, device=args.device, verbose=False)
        total_infer += time.time() - t0

        dets = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, clss, confs):
                dets.append((int(cls), float(conf), *box.tolist()))

        # 必须用素材自身的时间轴，不能用墙钟时间：
        # CPU 离线推理远慢于实时(如 5fps)，若按 time.time() 取 dt，
        # 同样的一帧位移会被摊到 5 倍长的时间里，算出的速度偏小 5 倍，运动事件永远触发不了。
        if kind == "video":
            msec = src.get(cv2.CAP_PROP_POS_MSEC)
            ts = (msec / 1000.0) if (msec and msec > 0) else (frame_idx / max(args.fps, 1))
        else:
            ts = frame_idx / max(args.fps, 1)

        tracks = tracker.update(dets, h, w, timestamp=ts)
        events = tracker.flush_events()
        for e in events:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        # 静态图没有时序，轨迹到不了 min_hits，必须把未确认的框也画出来
        out_frame = draw_tracks(
            frame.copy(), tracks, events, args.motion_threshold, include_inactive=(kind == "images")
        )

        if out_is_video:
            if writer is None:
                os.makedirs(Path(args.output).parent, exist_ok=True)
                out_h, out_w = h, w
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
            writer.write(out_frame)
        elif out_dir is not None:
            stem = Path(src[frame_idx]).stem if kind == "images" else f"{frame_idx:06d}"
            cv2.imwrite(str(out_dir / f"{stem}_motion.jpg"), out_frame)

        if args.view_img:
            cv2.imshow("MotionTracker", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            n_act = len([t for t in tracks if t.is_active])
            print(
                f"frame {frame_idx}, active={n_act}, avg_infer={total_infer / frame_idx * 1000:.1f}ms, "
                f"events(new/moving)={counts['new']}/{counts['moving']}"
            )

    if kind == "video":
        src.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(
        f"done: {frame_idx} frames, avg inference={total_infer / max(frame_idx, 1) * 1000:.1f}ms, "
        f"events: new={counts['new']} moving={counts['moving']}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/train/yolo11s_v2/weights/best.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--output", default="")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-nms", type=float, default=0.45)
    p.add_argument("--device", default="cpu", help="本地验证时建议 cpu，训练占用 gpu 期间不影响")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--view-img", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="0 表示处理全部，用于快速冒烟测试")
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-miss", type=int, default=5)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument(
        "--motion-threshold",
        type=float,
        default=0.10,
        help="归一化位移/秒(位移÷图像对角线÷秒)，1080p 下 0.10 ≈ 220 px/s",
    )
    p.add_argument("--min-motion-frames", type=int, default=3, help="连续超阈值帧数，滞回确认")
    p.add_argument("--no-compensate", action="store_true", help="关闭相机全局运动补偿")
    p.add_argument("--event-cooldown", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
