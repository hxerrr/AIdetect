"""RK3588 板端实时推理脚本：支持 USB 摄像头 / RTSP / 本地视频 / 图片序列

用法示例:
    # USB 摄像头
    python3 rknn/board_infer.py --model weights/yolo11s_aidetect_i8.rknn --source 0

    # RTSP 拉流
    python3 rknn/board_infer.py --model weights/yolo11s_aidetect_i8.rknn --source rtsp://admin:pass@ip:554/stream

    # 保存推理视频并叠加轨迹
    python3 rknn/board_infer.py --model weights/yolo11s_aidetect_i8.rknn --source /data/video.mp4 --output /data/out.mp4

注意:
    - 需在板端安装 rknn-toolkit-lite2。
    - 模型输入固定为 640x640; NPU 用 core_mask 默认三核。
"""

import argparse
import os
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "web"))
from rknn_postprocess import decode_yolo, letterbox, scale_coords  # noqa: E402
from tracker_common import MotionTracker, draw_tracks  # noqa: E402

try:
    from rknnlite.api import RKNNLite
except ImportError as e:
    print("[err] 未安装 rknn-toolkit-lite2，请在板端环境安装: pip install rknn-toolkit-lite2", e)
    raise


def ensure_db(db_path):
    """不依赖 flask/web_server，直接建表，便于板端单独落库"""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, timestamp REAL NOT NULL,
            cls INTEGER NOT NULL, label TEXT NOT NULL,
            track_id INTEGER, kind TEXT, conf REAL, speed REAL,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            image_path TEXT, pushed INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_ts ON events(timestamp);
        """
    )
    conn.commit()
    conn.close()


def emit_event(args, ws, evt, frame):
    """事件落库/推送：Web 开启时走队列，否则直接写 SQLite（表结构与 web_server 一致）

    evt 由 MotionTracker 产生，已包含 cls/label/track_id/kind/conf/speed/box，
    冷却与滞回确认都在 tracker 内部完成，这里只负责持久化。
    """
    from datetime import datetime

    evt = dict(evt)
    evt["ts"] = datetime.now().isoformat(timespec="seconds")
    evt["timestamp"] = evt.get("time", time.time())
    evt.setdefault("image_path", None)

    if args.snapshot_dir:
        Path(args.snapshot_dir).mkdir(parents=True, exist_ok=True)
        snap = Path(args.snapshot_dir) / f"{int(evt['timestamp'])}_{evt['label']}_{evt['track_id']}.jpg"
        cv2.imwrite(str(snap), frame)
        evt["image_path"] = str(snap)

    if ws is not None:
        try:
            ws.event_queue.put(evt, block=False)
            return
        except queue.Full:
            pass

    conn = sqlite3.connect(args.db)
    conn.execute(
        "INSERT INTO events (ts, timestamp, cls, label, track_id, kind, conf, speed, x1, y1, x2, y2, image_path)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            evt["ts"], evt["timestamp"], evt["cls"], evt["label"], evt["track_id"], evt["kind"],
            evt["conf"], evt["speed"], evt["x1"], evt["y1"], evt["x2"], evt["y2"], evt["image_path"],
        ),
    )
    conn.commit()
    conn.close()
    print(f"[event] {evt['label']}-{evt['track_id']} conf={evt['conf']} speed={evt['speed']} -> {args.db}")


def build_source(source):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise RuntimeError(f"摄像头打开失败: {source}")
        return cap, "camera"
    s = source.lower()
    if s.startswith("rtsp://"):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"RTSP 打开失败: {source}")
        return cap, "rtsp"
    p = Path(source)
    if p.is_dir():
        files = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
        return files, "images"
    if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
        cap = cv2.VideoCapture(str(p))
        return cap, "video"
    raise ValueError(f"不支持的 source: {source}")


def run(args):
    start = time.time()
    rknn = RKNNLite()
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        raise RuntimeError(f"load_rknn 失败: {ret}")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        raise RuntimeError(f"init_runtime 失败: {ret}")
    print(f"[rknn] loaded {args.model}")

    ws = None
    if args.web:
        import web_server
        web_server.DB_PATH = args.db
        web_server.init_db(args.db)
        threading.Thread(target=web_server.run_server, args=(args.db, args.host, args.port), daemon=True).start()
        ws = web_server
        print(f"[web] embedded server http://{args.host}:{args.port}")

    if args.save_events and ws is None:
        ensure_db(args.db)
        print(f"[db] events -> {args.db}")

    tracker = MotionTracker(
        iou_threshold=args.iou_threshold,
        max_miss=args.max_miss,
        min_hits=args.min_hits,
        motion_threshold=args.motion_threshold,
        event_cooldown=args.event_cooldown,
        min_motion_frames=args.min_motion_frames,
        default_fps=args.fps,
        compensate=not args.no_compensate,
    )

    src, kind = build_source(args.source)
    writer = None
    out_h, out_w = None, None
    frame_idx = 0
    t_infer = 0.0

    while True:
        if kind == "images":
            if frame_idx >= len(src):
                break
            frame = cv2.imread(str(src[frame_idx]))
            if frame is None:
                frame_idx += 1
                continue
        else:  # video / rtsp / camera
            ret, frame = src.read()
            if not ret or frame is None:
                if kind == "rtsp":
                    print("[warn] RTSP 读帧失败，重连...")
                    time.sleep(1)
                    src.release()
                    src = cv2.VideoCapture(args.source)
                    continue
                print(f"[warn] 读帧失败({kind})，结束")
                break

        if args.max_frames and frame_idx >= args.max_frames:
            break

        h0, w0 = frame.shape[:2]
        img_lb, scale, pad_left, pad_top = letterbox(frame, target_size=args.imgsz)
        blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        t0 = time.time()
        outs = rknn.inference(inputs=[blob])
        infer_ms = (time.time() - t0) * 1000
        t_infer += infer_ms

        dets_lb = decode_yolo(outs[0], conf_thresh=args.conf, iou_thresh=args.iou_nms)
        dets = []
        for cls_id, conf, x1, y1, x2, y2 in dets_lb:
            arr = scale_coords([[x1, y1, x2, y2]], (h0, w0), input_size=args.imgsz, pad_left=pad_left, pad_top=pad_top, scale=scale)
            nx1, ny1, nx2, ny2 = arr[0]
            dets.append((cls_id, conf, float(nx1), float(ny1), float(nx2), float(ny2)))

        tracks = tracker.update(dets, h0, w0)
        active = [t for t in tracks if t.is_active]
        events = tracker.flush_events()
        vis = draw_tracks(frame.copy(), tracks, events, args.motion_threshold)

        # FPS 信息
        fps = frame_idx / (time.time() - start + 1e-6) if frame_idx else 0
        info = f"frame={frame_idx} fps={fps:.1f} infer={infer_ms:.1f}ms active={len(active)}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 推送实时帧与统计；运动事件已由 tracker 完成滞回确认与冷却，这里只落库
        if ws is not None:
            ws.set_frame(vis)
            ws.set_stats({"fps": round(fps, 1), "infer_ms": round(infer_ms, 1), "active": len(active)})
        if ws is not None or args.save_events:
            for e in events:
                if e["kind"] == "moving":
                    emit_event(args, ws, e, vis)

        if writer is None and args.output:
            os.makedirs(Path(args.output).parent, exist_ok=True)
            out_h, out_w = h0, w0
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
        if writer:
            if (h0, w0) != (out_h, out_w):
                vis = cv2.resize(vis, (out_w, out_h))
            writer.write(vis)

        if args.show:
            cv2.imshow("AIdetect-RK3588", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[info] {info} avg_infer={t_infer/frame_idx:.1f}ms")

    if kind in ("video", "rtsp", "camera"):
        src.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    rknn.release()
    print(f"[done] total frames={frame_idx} avg_infer={t_infer/max(frame_idx,1):.1f}ms")


if __name__ == "__main__":
    start = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", default="")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-nms", type=float, default=0.45)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--show", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="0 表示不限，用于快速验证")
    p.add_argument("--web", action="store_true", help="同时启动 Flask Web 服务")
    p.add_argument("--save-events", action="store_true", help="不开 Web 也把事件写入 SQLite")
    p.add_argument("--snapshot-dir", default="", help="事件触发时保存现场快照的目录")
    p.add_argument("--db", default="web/aidetect.db")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
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
    args = p.parse_args()
    run(args)
