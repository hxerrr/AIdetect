"""拉取多路视频流(RTSP/本地文件) -> YOLOv8s_v1 实时检测 -> Web 多画面可视化

单进程，每一路各一组线程，互不阻塞：
  - 捕获线程 x N : RTSP 拉流(FFMPEG/TCP、低缓冲)，断流自动重连，只保留最新一帧
  - 推理线程 x N : 各自加载一份 YOLO 权重 + 独立 MotionTracker(轨迹/事件互不串)
  - 主线程      : 起 Flask(web/web_server.py)，/video_feed?cam=N 按通道出流，页面自动排宫格

海康 RTSP 地址格式:
    rtsp://<用户>:<密码>@<IP>:554/Streaming/Channels/<通道><码流>
    101 = 通道1 主码流(1080P/4K，清晰但吃带宽和算力)
    102 = 通道1 子码流(720P/D1，实时检测推荐用这个)
    球机多通道时把首位换成通道号，如 201/202。

用法:
    # 单路球机(子码流，推荐)
    python tools/rtsp_web_detect.py --rtsp "rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/102"

    # 多路同测：--rtsp / --source 可重复传，顺序即通道号 0/1/2...
    python tools/rtsp_web_detect.py \
        --rtsp "rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/102" \
        --rtsp "rtsp://admin:12345@192.168.1.65:554/Streaming/Channels/102" \
        --source video/input/collapse01.mp4

    # 主码流 + 1280 推理分辨率
    python tools/rtsp_web_detect.py --rtsp "rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101" --imgsz 1280

    # 浏览器打开 http://<本机IP>:5000
"""

import argparse
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

import cv2

ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / "web" / "snapshots"          # 告警截图存放目录
EVT_DIR = ROOT / "dataset" / "images" / "train"  # 事件原始帧存这里，给回训当负样本
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "tools"))

import web_server                                   # noqa: E402
from tracker_common import MotionTracker, draw_tracks  # noqa: E402

# web_server 里的默认路径是相对 CWD 的，这里改成绝对，避免换目录启动后找不到库
web_server.DB_PATH = str(ROOT / "web" / "aidetect.db")
web_server.CONFIG_PATH = str(ROOT / "web" / "config.json")

DEFAULT_WEIGHTS = "runs/train/yolov8s_v1/weights/best.pt"


def mask_pwd(url):
    """日志里打 URL 时不暴露密码: rtsp://user:***@host/..."""
    s = str(url)
    if not s.lower().startswith("rtsp://") or "@" not in s:
        return s
    head, _, tail = s[7:].rpartition("@")     # 主机里不会有 @，最后一个 @ 才是分隔符
    user = head.split(":", 1)[0]
    return f"rtsp://{user}:***@{tail}"


def rtsp_variants(url):
    """同一条流返回几种写法，逐个试。

    海康密码常带 @ / # / % 这类字符，而 RTSP URL 里 @ 又是"凭据结束"的分隔符：
    - 写明文 @ : ffmpeg 会把 2025@192.168.110.86 当成主机名，解析直接错
    - 写 %40   : 部分 ffmpeg 版本不做 percent 解码，密码变成字面量 Yunfan%402025
    两种都不保证成功，所以编码/解码两个方向都试一遍，哪个能连上用哪个。
    """
    s = str(url)
    PFX = "rtsp://"
    if not s.lower().startswith(PFX):
        return [s]
    rest = s[len(PFX):]
    authhost, sep, path = rest.partition("/")
    out = [s]
    if "@" in authhost:
        auth, _, host = authhost.rpartition("@")
        user, _, pwd = auth.partition(":")
        if pwd:
            for alt in (quote(pwd, safe=""), unquote(pwd)):
                if alt != pwd:
                    out.append(f"{PFX}{user}:{alt}@{host}{sep}{path}")
    return out


def open_capture(source, tcp=True):
    """打开 RTSP / 视频文件 / 摄像头。

    RTSP 必须走 TCP：UDP 在丢包时 OpenCV 会卡在 read() 上几十秒才返回。
    stimeout/max_delay 单位是微秒，给的是"连不上/收不到数据"的超时，避免断网后永久阻塞。
    """
    s = str(source)
    if s.isdigit():
        cap = cv2.VideoCapture(int(s))
    elif s.lower().startswith("rtsp://"):
        # OPENCV_FFMPEG_CAPTURE_OPTIONS 是 OpenCV 读的全局环境变量，必须在 open 之前设置
        if tcp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"
            )
        cap = None
        for u in rtsp_variants(s):
            c = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
            if c.isOpened():
                if u != s:
                    print(f"[cap] 凭据换用 {mask_pwd(u)} 才连上", flush=True)
                cap = c
                break
            c.release()
        if cap is None:                      # 都失败也返回一个对象，让上层看到 isOpened()=False
            cap = cv2.VideoCapture(s, cv2.CAP_FFMPEG)
        # 缓冲压到 1 帧，否则播放器拿到的是几秒前的画面
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(s)
    return cap


def pick_device(dev):
    try:
        import torch
    except ImportError:
        return "cpu"
    if str(dev).isdigit():
        if torch.cuda.is_available() and int(dev) < torch.cuda.device_count():
            return str(dev)
        print("[warn] CUDA 不可用，回退 CPU")
        return "cpu"
    return str(dev)


def capture_worker(cam_id, source, frame_q, stop_ev, tcp=True):
    """捕获线程：只保留最新一帧，断流自动重连"""
    tag = f"[cam {cam_id}]"
    # 单独一路连不上(密码错/通道号不对/设备离线)不能连坐其它通道，
    # 所以这里只重试、不 set(stop_ev)，起来后页面上也只是这一路没画面。
    cap = None
    miss_open = 0
    while not stop_ev.is_set():
        cap = open_capture(source, tcp=tcp)
        if cap.isOpened():
            break
        cap.release()
        cap = None
        miss_open += 1
        print(f"{tag}[err] 打开失败({miss_open})，5s 后重试: {mask_pwd(source)}", flush=True)
        time.sleep(5)
    if cap is None:
        print(f"{tag}[err] 放弃该路: {mask_pwd(source)}", flush=True)
        return
    print(f"{tag}[cap] 已连接 {mask_pwd(source)}", flush=True)

    # 本地文件按原帧率节流：否则读盘远快于推理，几秒就把整段视频灌完，根本测不出东西
    # RTSP / 摄像头本身由码流节奏限速，不能节流
    s = str(source)
    is_file = not (s.isdigit() or s.lower().startswith("rtsp://"))
    src_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
    pace = 1.0 / src_fps if is_file and 1.0 < src_fps < 120.0 else 0.0
    if pace:
        print(f"{tag}[cap] 本地文件按 {src_fps:.1f} FPS 节流")

    miss = 0
    while not stop_ev.is_set():
        t_beg = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            miss += 1
            is_live = s.lower().startswith("rtsp://") or s.isdigit()
            if not is_live:      # 本地视频读到底
                print(f"{tag}[cap] 视频结束", flush=True)
                break
            print(f"{tag}[warn] 读帧失败({miss})，重连中...", flush=True)
            cap.release()
            time.sleep(min(2 * miss, 10))
            cap = open_capture(source, tcp=tcp)
            if not cap.isOpened():
                continue
            print(f"{tag}[cap] 重连成功", flush=True)
            continue
        miss = 0
        # 队列只留最新帧：推理慢时宁可丢帧也不要堆积造成延迟
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put(frame)
        if pace:
            dt = time.time() - t_beg
            if dt < pace:
                time.sleep(pace - dt)
    cap.release()
    print(f"{tag}[cap] 退出", flush=True)


def save_snapshot(frame, e, cam_id):
    """告警截图: 连同检测框一起存，返回相对 web 根的路径(供前端 <img> 直接引)"""
    sub = SNAP_DIR / f"cam{cam_id}"
    sub.mkdir(parents=True, exist_ok=True)
    name = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{e.get('label','event')}"
            f"_id{e.get('track_id',0)}_{int(e.get('conf',0)*100)}.jpg")
    ok = cv2.imwrite(str(sub / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return f"snapshots/cam{cam_id}/{name}" if ok else ""


def to_web_event(e, cam_id=0):
    """把 tracker 事件补成 web_server 落库需要的字段"""
    now = time.time()
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "timestamp": now,
        "cam": cam_id,
        "cls": e["cls"],
        "label": e["label"],
        "track_id": e["track_id"],
        "kind": e["kind"],
        "conf": e["conf"],
        "speed": e["speed"],
        "x1": e["x1"], "y1": e["y1"], "x2": e["x2"], "y2": e["y2"],
    }


def infer_worker(cam_id, source, frame_q, stop_ev, args, device):
    """推理线程：每路独立加载一份权重 + 独立跟踪器，避免多路共用模型/轨迹互相干扰"""
    tag = f"[cam {cam_id}]"
    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names
    print(f"{tag}[infer] 权重 {args.weights}  源 {source}", flush=True)

    tracker = MotionTracker(
        motion_threshold=args.motion_threshold,
        event_cooldown=args.event_cooldown,
        static_after=args.static_after,
        fixture_ttl=args.fixture_ttl,
        names=names,
        min_event_hits=args.confirm_hits,
        min_disp_ratio=args.min_disp_ratio,
    )
    if cam_id == 0:
        print(f"[infer] 固定物抑制: 静止超过 {args.static_after}s 不再进事件表"
              if args.static_after > 0 else "[infer] 固定物抑制: 关闭", flush=True)

    fps = 0.0
    frame_i = 0
    last_t = time.time()
    last_evt_save = 0.0                        # 上次存事件帧的时刻(节流用)
    edge_drop_n, edge_log_t = 0, time.time()   # 边缘排除计数/上次打印时刻

    while not stop_ev.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        frame_i += 1

        t0 = time.time()
        # half 已被 ultralytics 标记废弃，传 False 也会刷警告，因此仅在显式开启时传
        kw = {"half": True} if args.half else {}
        res = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=device, verbose=False, **kw
        )[0]
        infer_ms = (time.time() - t0) * 1000.0

        h, w = frame.shape[:2]
        dets = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                dets.append((int(b.cls[0]), float(b.conf[0]), x1, y1, x2, y2))

        # 面积双阈值过滤(进跟踪器之前)：室内反光地面/灯光光斑常被整片框成滑坡，
        # 真实滚石滑坡在这个镜头距离下不会占满 30% 画面；过小的则是噪点
        frame_area = float(h * w) or 1.0
        dets = [d for d in dets
                if args.min_area_ratio * frame_area
                <= (d[4] - d[2]) * (d[5] - d[3]) <= args.max_area_ratio * frame_area]

        # 边缘近景排除：贴着画面边框、又足够大的框，基本都是镜头前的失焦遮挡物
        # (手、肩膀、镜头脏污、门框)，不是远处的落石。小目标不误杀，仍可从边缘正常进入。
        if args.edge_margin > 0 and dets:
            mx, my = w * args.edge_margin, h * args.edge_margin
            keep = []
            for d in dets:
                _, _, x1, y1, x2, y2 = d
                near = (x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my)
                big = (x2 - x1) * (y2 - y1) >= args.edge_area * frame_area
                if not (near and big):
                    keep.append(d)
            dropped = len(dets) - len(keep)
            if dropped:
                edge_drop_n += dropped
                # 这种框每帧都会出现，逐帧打印会把日志刷爆，10s 汇总一次
                if time.time() - edge_log_t >= 10:
                    print(f"{tag}[filter] 边缘近景 10s 内排除 {edge_drop_n} 个", flush=True)
                    edge_drop_n, edge_log_t = 0, time.time()
            dets = keep

        tracks = tracker.update(dets, h, w)
        events = tracker.flush_events()

        # 触发事件的原始帧(未画框)存进训练集，前缀 evt：
        # 告警截图(save_snapshot)是画了框给网页看的，不能直接拿去训练；
        # 这里补一份干净帧。这些"模型报了事件"的画面正是回训最有价值的样本
        # (当前场景全是误检 -> 当负样本用；将来出现真目标后，
        #  记得把对应的 evt*.jpg 删掉或挪去标注，否则真落石会被当背景训掉)。
        # 节流: 同一路 evt-save-interval 秒内只存一张，new/moving 同帧成对触发也只存一次。
        now = time.time()
        if events and args.evt_save_interval > 0 and now - last_evt_save >= args.evt_save_interval:
            last_evt_save = now
            e0 = max(events, key=lambda x: x.get("conf", 0))
            EVT_DIR.mkdir(parents=True, exist_ok=True)
            name = (f"evtcam{cam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    f"_{e0.get('label', 'event')}_{int(e0.get('conf', 0) * 100)}.jpg")
            cv2.imwrite(str(EVT_DIR / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            print(f"[cam{cam_id}] 事件帧已存训练集: {name}", flush=True)

        # 先画框再截图，截图里带检测框(状态条在截图之后画，避免截到 cam 横幅)
        # 注意 draw_tracks 的签名是 (frame, tracks, ...)，别写反
        draw_tracks(frame, tracks, events,
                    motion_threshold=args.motion_threshold,
                    show_trail=args.trail,
                    include_inactive=args.show_all,
                    min_disp_ratio=args.min_disp_ratio)

        # 推事件进 web 的落库队列(同进程，直接 put，不走 HTTP)，有告警先存截图
        for e in events:
            try:
                evt = to_web_event(e, cam_id)
                evt["image_path"] = save_snapshot(frame, e, cam_id)
                web_server.event_queue.put_nowait(evt)
            except queue.Full:
                pass

        # 叠加实时状态条(带通道号，多路时能对上是哪一路)
        cv2.rectangle(frame, (0, 0), (340, 28), (0, 0, 0), -1)
        cv2.putText(frame, f"cam{cam_id} {args.weights.split('/')[-2]} | {len(dets)} obj | {infer_ms:.0f}ms",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        web_server.set_frame(frame, cam_id)

        dt = time.time() - last_t
        last_t = time.time()
        if dt > 0:
            cur = 1.0 / dt
            fps = cur if fps == 0 else fps * 0.9 + cur * 0.1
        if frame_i % 100 == 0:
            print(f"{tag}[infer] frame={frame_i} fps={fps:.1f} infer={infer_ms:.0f}ms "
                  f"dets={len(dets)} tracks={len(tracks)} events={len(events)}", flush=True)

        web_server.set_stats({
            "source": str(source),
            "fps": round(fps, 1),
            "infer_ms": round(infer_ms, 1),
            # 活跃目标只数"会告警的"，固定物不计入
            "active": sum(1 for t in tracker.active_tracks()
                          if not getattr(t, "is_fixture", False)),
        }, cam_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", action="append", default=None,
                    help="海康 RTSP 地址，可重复传以接入多路(顺序即通道号)")
    ap.add_argument("--source", action="append", default=None,
                    help="本地视频/摄像头，可重复传，与 --rtsp 混用(调试用)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="实时检测用 0.25，别用评估的 0.001")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true", help="FP16 推理(GPU 上约快 1.5x)")
    ap.add_argument("--motion-threshold", type=float, default=0.10)
    ap.add_argument("--min-area-ratio", type=float, default=0.0005,
                    help="检测框最小面积占整帧比例，低于此按噪点丢弃")
    ap.add_argument("--max-area-ratio", type=float, default=0.30,
                    help="检测框最大面积占整帧比例，超过说明是整片地面/反光被框住，丢弃")
    ap.add_argument("--event-cooldown", type=float, default=2.0)
    ap.add_argument("--evt-save-interval", type=float, default=60.0,
                    help="触发事件的原始帧存训练集的节流间隔(秒)，0=关闭。"
                         "事件帧是最有价值的回训样本；当前全是误检场景直接当负样本用")
    ap.add_argument("--confirm-hits", type=int, default=0,
                    help="事件确认帧数: 轨迹连续命中这么多帧才允许报事件(0=沿用 min_hits=3)。"
                         "8 能挡掉地面反光/灯光光斑这类只闪几帧的误检")
    ap.add_argument("--min-disp-ratio", type=float, default=0.0,
                    help="净位移门槛(占图像对角线比例): 中心点从轨迹出生起没挪够这么远就不算 Moving。"
                         "0=关闭; 0.03 约等于 640x360 下 22px，专门治'静止石头被框抖动报 Moving'")
    ap.add_argument("--edge-margin", type=float, default=0.0,
                    help="边缘排除: 检测框贴边(距画面边缘 < 该比例)且够大时才丢弃，0=关闭。"
                         "挡镜头前的失焦近景/遮挡物(0.06 约为画面 6%%)")
    ap.add_argument("--edge-area", type=float, default=0.10,
                    help="边缘排除的面积门槛: 贴边且面积占整帧 >= 该比例才丢，避免误杀从边缘正常进入的目标")
    ap.add_argument("--static-after", type=float, default=5.0,
                    help="静止超过该秒数判定为固定物(背景石头)，不再进事件表；0=关闭抑制")
    ap.add_argument("--fixture-ttl", type=float, default=60.0,
                    help="固定物记忆时长(秒)，防止轨迹断 ID 重新检出后又报一次")
    ap.add_argument("--show-all", action="store_true", default=True,
                    help="画出所有检测框(默认开)；只画已确认轨迹用 --no-show-all")
    ap.add_argument("--no-show-all", dest="show_all", action="store_false")
    ap.add_argument("--no-trail", dest="trail", action="store_false", help="不画运动轨迹线")
    ap.add_argument("--udp", action="store_true", help="RTSP 用 UDP(默认 TCP，UDP 丢包会卡死)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    # --rtsp / --source 均可重复，合并成一个源列表，下标就是通道号
    sources = list(args.rtsp or []) + list(args.source or [])
    if not sources:
        raise SystemExit(
            "[err] 需要视频源。例:\n"
            '  --rtsp "rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/102"\n'
            "  多路: --rtsp <url1> --rtsp <url2> --source video/input/collapse01.mp4"
        )

    device = pick_device(args.device)
    print(f"[dev] 推理设备: {device}  路数: {len(sources)}")

    stop_ev = threading.Event()
    threads = []
    for cam_id, source in enumerate(sources):
        frame_q = queue.Queue(maxsize=1)   # 每路独立帧队列
        threads.append(threading.Thread(
            target=capture_worker, args=(cam_id, source, frame_q, stop_ev, not args.udp),
            daemon=True, name=f"cap{cam_id}"))
        threads.append(threading.Thread(
            target=infer_worker, args=(cam_id, source, frame_q, stop_ev, args, device),
            daemon=True, name=f"infer{cam_id}"))
    for t in threads:
        t.start()

    print(f"[web] http://127.0.0.1:{args.port}  共 {len(sources)} 路  (Ctrl+C 退出)")
    try:
        web_server.run_server(web_server.DB_PATH, args.host, args.port)
    except KeyboardInterrupt:
        stop_ev.set()


if __name__ == "__main__":
    main()
