"""AIdetect Web 服务：事件接收、存储、推送、实时视频流、可视化

用法:
    # 在板端与 board_infer.py 一起启动
    python web/web_server.py --db web/aidetect.db --port 5000

接口:
    POST /api/event       接收检测事件(JSON)
    GET  /api/events      事件列表(支持 ?limit=50&cls=&start=)
    GET  /video_feed      MJPEG 实时视频流
    GET  /                可视化首页

推送:
    在 web/config.json 中配置 SMTP/企业微信/钉钉 webhook，事件触发时自动推送。
"""

import argparse
import json
import queue
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")

# 多路视频帧缓存: cam_id -> 最新一帧的 JPEG 字节(编码一次，多个观看端复用)
# 单路时 cam_id=0，与旧版 set_frame(frame) 的调用方式兼容
# 注意: 必须是 Lock() 实例。写成 threading.Lock(类) 时每次 with 都会新建一把锁，等于没加锁。
frames_lock = threading.Lock()
FRAMES = {}
event_queue = queue.Queue(maxsize=1000)

# 运行时统计(FPS / 推理耗时 / 活跃目标): cam_id -> dict，由各路推理线程每帧写入
stats_lock = threading.Lock()
STATS = {}

# 推送(邮件/微信/钉钉)走独立线程池，避免 SMTP 超时阻塞事件落库线程
push_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="push")

DB_PATH = "web/aidetect.db"
CONFIG_PATH = "web/config.json"
# 告警截图根目录(相对项目根，库里存相对路径，换机器不失效)
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"


def init_db(db_path):
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            timestamp REAL NOT NULL,
            cls INTEGER NOT NULL,
            label TEXT NOT NULL,
            track_id INTEGER,
            kind TEXT,
            conf REAL,
            speed REAL,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            image_path TEXT,
            cam INTEGER DEFAULT 0,
            pushed INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_ts ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_cls ON events(cls);
        """
    )
    # 旧库没有 cam 列时补上(多路化之前的记录都归到 0 号通道)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()]
    if "cam" not in cols:
        conn.execute("ALTER TABLE events ADD COLUMN cam INTEGER DEFAULT 0")
    conn.commit()
    conn.close()
    print(f"[db] initialized {db_path}")


def load_config():
    if not Path(CONFIG_PATH).exists():
        return {"smtp": None, "wechat": None, "dingtalk": None}
    return json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))


def push_event(evt, cfg):
    """事件推送：邮件/企业微信/钉钉，失败不阻塞"""
    text = f"AIdetect 预警[通道{evt.get('cam', 0)}]: {evt['label']} | {evt['kind']} | conf={evt.get('conf',0):.2f}"
    # SMTP
    smtp = cfg.get("smtp")
    if smtp:
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(json.dumps(evt, ensure_ascii=False, indent=2), "plain", "utf-8")
            msg["Subject"] = text
            msg["From"] = smtp["user"]
            msg["To"] = ",".join(smtp["to"])
            with smtplib.SMTP_SSL(smtp["host"], smtp["port"]) as s:
                s.login(smtp["user"], smtp["pass"])
                s.sendmail(smtp["user"], smtp["to"], msg.as_string())
            print("[push] email sent")
        except Exception as e:
            print("[push] email failed:", e)
    # 企业微信 webhook
    wx = cfg.get("wechat")
    if wx:
        try:
            import urllib.request

            data = json.dumps({"msgtype": "text", "text": {"content": text}}, ensure_ascii=False).encode()
            req = urllib.request.Request(wx["webhook"], data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            print("[push] wechat sent")
        except Exception as e:
            print("[push] wechat failed:", e)
    # 钉钉 webhook
    dd = cfg.get("dingtalk")
    if dd:
        try:
            import urllib.request

            data = json.dumps({"msgtype": "text", "text": {"content": text}}, ensure_ascii=False).encode()
            req = urllib.request.Request(dd["webhook"], data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            print("[push] dingtalk sent")
        except Exception as e:
            print("[push] dingtalk failed:", e)


def event_worker(db_path):
    """后台线程：持久化事件并触发推送"""
    cfg = load_config()
    while True:
        try:
            evt = event_queue.get(timeout=1)
        except queue.Empty:
            continue
        if evt is None:
            break
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO events (ts, timestamp, cls, label, track_id, kind, conf, speed, x1, y1, x2, y2, image_path, cam)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evt.get("ts"),
                evt.get("timestamp"),
                evt.get("cls"),
                evt.get("label"),
                evt.get("track_id"),
                evt.get("kind"),
                evt.get("conf"),
                evt.get("speed"),
                evt.get("x1"),
                evt.get("y1"),
                evt.get("x2"),
                evt.get("y2"),
                evt.get("image_path"),
                evt.get("cam", 0),
            ),
        )
        conn.commit()
        evt["id"] = cur.lastrowid
        conn.close()
        # 推送放线程池：SMTP/webhook 超时不应拖慢事件落库
        push_pool.submit(push_event, evt, cfg)


def set_frame(frame, cam=0):
    """写入某一路的最新帧(编码成 JPEG bytes，多个观看端复用同一份，不再每客户端重编码)"""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return
    with frames_lock:
        FRAMES[int(cam)] = buf.tobytes()


def get_frame_bytes(cam=0):
    with frames_lock:
        return FRAMES.get(int(cam))


def cams():
    """当前有画面的通道号"""
    with frames_lock:
        return sorted(FRAMES.keys())


def set_stats(d, cam=0):
    """更新某一路的运行时统计，供 /api/stats 与前端卡片使用"""
    with stats_lock:
        STATS.setdefault(int(cam), {}).update(d)


def get_stats(cam=None):
    """cam=None 返回所有路 {cam_id: {...}}，否则只返回该路"""
    with stats_lock:
        if cam is None:
            return {str(k): dict(v) for k, v in STATS.items()}
        return dict(STATS.get(int(cam), {}))


def mjpeg_stream(cam=0):
    while True:
        buf = get_frame_bytes(cam)
        if buf is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/event", methods=["POST"])
def api_event():
    evt = request.get_json(force=True)
    evt["ts"] = datetime.now().isoformat(timespec="seconds")
    evt["timestamp"] = time.time()
    try:
        event_queue.put(evt, block=False)
    except queue.Full:
        return jsonify({"ok": False, "error": "event queue full"}), 503
    return jsonify({"ok": True})


@app.route("/api/events")
def api_events():
    limit = max(1, min(500, int(request.args.get("limit", 50))))
    cls_filter = request.args.get("cls", "")
    start = request.args.get("start", "")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    sql = "SELECT * FROM events WHERE 1=1"
    params = []
    if cls_filter:
        sql += " AND cls=?"
        params.append(int(cls_filter))
    if start:
        sql += " AND timestamp>=?"
        params.append(float(start))
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in conn.execute(sql, params)]
    conn.close()
    return jsonify({"events": rows})


@app.route("/api/events", methods=["DELETE"])
def api_events_clear():
    """一键清除全部事件(含截图文件，避免磁盘只进不出)"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT image_path FROM events")
    for (img,) in cur.fetchall():
        if img:
            f = Path(__file__).resolve().parent.parent / img
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass        # 删不掉的文件不阻塞清库
    n = conn.execute("DELETE FROM events").rowcount
    conn.commit()
    conn.close()
    removed = n or 0
    print(f"[db] 清除事件 {removed} 条", flush=True)
    return jsonify({"ok": True, "removed": removed})


@app.route("/snapshots/<path:fname>")
def snapshots(fname):
    """告警截图静态访问: /snapshots/cam0/20260920_164507_rockfall_id7_93.jpg"""
    return send_from_directory(SNAPSHOT_DIR, fname)


@app.route("/api/cams")
def api_cams():
    return jsonify({"cams": cams()})


@app.route("/api/stats")
def api_stats():
    # 不带 cam 参数返回所有路 {cam_id:{fps,infer_ms,active}}，带 cam 只返回该路
    cam = request.args.get("cam", "")
    return jsonify(get_stats(None if cam == "" else int(cam)))


@app.route("/video_feed")
def video_feed():
    cam = int(request.args.get("cam", 0))
    return Response(mjpeg_stream(cam), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------- 合成九宫格单流 ----------------
# 浏览器对同一 host:port 的 HTTP/1.1 并发连接上限约 6 个，9 路独立视频流会把
# 名额占满，/api/stats 轮询和"一键清除"就会无限排队(页面显示连接断开、清理无效)。
# 所以只开一路 MJPEG：服务器把各路最新帧拼成 3x3 大图，页面 1 个 <img> 搞定。
GRID_COLS = 3
GRID_ROWS = 3
CELL_W, CELL_H = 640, 360       # 3x3 拼完正好 1920x1080


def build_grid_jpeg():
    canvas = np.zeros((CELL_H * GRID_ROWS, CELL_W * GRID_COLS, 3), dtype=np.uint8)
    with frames_lock:
        snap = dict(FRAMES)          # cam_id -> 最新一帧 JPEG 字节
    for idx in range(GRID_COLS * GRID_ROWS):
        buf = snap.get(idx)
        if buf is None:              # 未接入的槽位画占位，别整块死黑
            x = (idx % GRID_COLS) * CELL_W
            y = (idx // GRID_COLS) * CELL_H
            cv2.putText(canvas, f"CAM {idx}", (x + CELL_W // 2 - 80, y + CELL_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 70, 90), 2, cv2.LINE_AA)
            continue
        img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, GRID_COLS)
        canvas[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = img
    ok, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpg.tobytes() if ok else None


@app.route("/video_grid")
def video_grid():
    def gen():
        while True:
            jpg = build_grid_jpeg()
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1 / 15)        # 合成流 15fps 足够看
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def run_server(db_path, host, port):
    init_db(db_path)
    t = threading.Thread(target=event_worker, args=(db_path,), daemon=True)
    t.start()
    print(f"[web] starting http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="web/aidetect.db")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    run_server(args.db, args.host, args.port)
