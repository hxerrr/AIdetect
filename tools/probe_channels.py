"""扫描海康 RTSP 通道：确认新接的摄像头该用哪个通道号/路径。

用法:
    python tools/probe_channels.py --host 192.168.110.210 --host 192.168.110.214
    python tools/probe_channels.py --host 192.168.110.86 --user admin --password 'Yunfan@2025'

对每个 IP 逐个试候选通道号，打印是否可连、分辨率，并保存一张缩略图到
logs/probe_thumbs/ 便于肉眼确认这是哪个镜头。密码里的 @ 两种写法都会试。
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
THUMB = ROOT / "logs" / "probe_thumbs"

# 海康常见路径: x01=主码流, x02=子码流; 独立摄像机一般只有 1xx/2xx，NVR 才有 4xx
CANDIDATES = [
    "Streaming/Channels/101", "Streaming/Channels/102",
    "Streaming/Channels/201", "Streaming/Channels/202",
    "Streaming/Channels/301", "Streaming/Channels/302",
    "Streaming/Channels/401", "Streaming/Channels/402",
    "Streaming/Channels/501", "Streaming/Channels/502",
    "Streaming/Channels/1", "Streaming/Channels/2",
]


def env_tcp():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;3000000|max_delay;300000"
    )


def url_variants(base, user, pwd):
    """@ 在 RTSP 里是凭据分隔符，密码含 @ 时编码/明文两种都试"""
    from urllib.parse import quote, unquote
    out = []
    for p in dict.fromkeys([pwd, quote(pwd, safe=""), unquote(pwd)]):
        out.append(f"rtsp://{user}:{p}@{base}")
    return out


def try_open(url, timeout_s=6.0):
    env_tcp()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return False, None, None
    t0 = time.time()
    frame = None
    while time.time() - t0 < timeout_s:
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            break
    if frame is None:
        cap.release()
        return True, None, None           # 能开但取不到帧(多半是流未启/编码问题)
    h, w = frame.shape[:2]
    cap.release()
    return True, frame, (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="append", required=True)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="Yunfan@2025")
    ap.add_argument("--port", type=int, default=554)
    ap.add_argument("--channels", default=",".join(CANDIDATES))
    args = ap.parse_args()

    THUMB.mkdir(parents=True, exist_ok=True)
    chans = [c.strip() for c in args.channels.split(",") if c.strip()]
    ok_list = []

    for host in args.host:
        base = f"{host}:{args.port}"
        print(f"\n=== {host} ===", flush=True)
        for ch in chans:
            full = f"{base}/{ch}"
            good = False
            for u in url_variants(full, args.user, args.password):
                opened, frame, size = try_open(u)
                if opened and frame is not None:
                    tag = f"{host}_{ch.replace('/', '_')}"
                    p = THUMB / f"{tag}.jpg"
                    cv2.imwrite(str(p), frame)
                    print(f"  [OK]  /{ch}  {size[0]}x{size[1]}  -> {p}", flush=True)
                    ok_list.append(f"rtsp://{args.user}:{args.password}@{full}")
                    good = True
                    break
                if opened:
                    print(f"  [??]  /{ch}  能开但取不到帧", flush=True)
                    good = True
                    break
            if not good:
                print(f"  [--]  /{ch}  连不上", flush=True)

    print("\n=== 可用地址(可直接粘进 start_live.ps1) ===", flush=True)
    for u in ok_list:
        print(u, flush=True)
    if not ok_list:
        print("(一个都没通：检查 IP/密码/554 端口，或用 VLC 试一条)", flush=True)


if __name__ == "__main__":
    env_tcp()
    sys.exit(main())
