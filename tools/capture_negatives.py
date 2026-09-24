"""从 RTSP 监控采集负样本(背景图)，用于压制室内反光地面这类误检。

原理: Ultralytics 训练时，没有对应 label 文件的图片按"纯背景"处理。
把误检场景(反光地面/灯光光斑)的帧定期存进 dataset/images/train，
回训后模型会学会"这种画面里没有滑坡/落石"。

用法:
    # 两路监控各采 200 张，每 20s 一张，第 10 张抽 1 张进验证集
    python tools/capture_negatives.py --count 200 --interval 20

    # 自定义源/输出
    python tools/capture_negatives.py --rtsp <url1> --rtsp <url2> --count 200

注意:
  - 采集时段尽量覆盖不同光照(白天/傍晚/开灯/关灯)，可以分几天各跑一轮；
  - 别一次灌太多: 全是同一场景的负样本会稀释正样本，单场景 200-400 张足够；
  - 采集的图不带 label，就是靠"没有 label"表达背景语义，不要手工补标注；
  - 默认会先跑一遍当前权重，画面里检出目标(置信度 >= --filter-conf)就跳过不存：
    否则把"桌上摆着石头"的画面当纯背景训进去，等于教模型"石头也是背景"，
    会把真石头一起学没。确认某路画面里确实没有可检目标时才用 --no-filter。
"""
import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def open_rtsp(url):
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000|max_delay;500000",
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def looks_same(a, b, thresh=6.0):
    """粗查重: 均值哈希差值小于阈值视为近似重复帧，避免存一堆一样的图"""
    if a is None or b is None:
        return False
    ah = cv2.resize(a, (16, 16)).mean(axis=2)
    bh = cv2.resize(b, (16, 16)).mean(axis=2)
    return float(abs(ah - bh).mean()) < thresh


def _hm_to_min(s):
    """"18:30" -> 1110（当天 0 点起的分钟数）"""
    try:
        h, m = str(s).split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception:
        return 0


def force_gap_now(args):
    """按当前时刻返回强制存图间隔(秒) + 是否夜间。

    白天 5 分钟一张，18:30 到次日 06:00 拉长到 20 分钟一张：夜里画面基本不变
    (灯恒定、没人走动)，采得再密也只是重复图，反而把白天的样本比例挤掉。
    """
    start, end = _hm_to_min(args.night_start), _hm_to_min(args.night_end)
    now = datetime.datetime.now()
    hm = now.hour * 60 + now.minute
    if start > end:                       # 跨零点，如 18:30 -> 06:00
        night = hm >= start or hm < end
    else:
        night = start <= hm < end
    gap = args.night_minutes if night else args.force_minutes
    return max(gap, 0.0) * 60.0, night


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", action="append", default=[
        "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/702",
        "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/402",
        "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/202",
        "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/302",
        "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/502",
        "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/602",
    ], help="RTSP 地址，可重复；默认就是当前六路监控")
    ap.add_argument("--count", type=int, default=200, help="每路采集张数")
    ap.add_argument("--interval", type=float, default=20.0, help="抽帧间隔(秒)")
    ap.add_argument("--force-minutes", type=float, default=5.0,
                    help="白天: 画面一直没变化也强制存一张的间隔(分钟)。静止场景靠'画面变化'判定"
                         "永远存不满，必须按时间兜底才能覆盖光照变化")
    ap.add_argument("--night-minutes", type=float, default=20.0,
                    help="夜间(--night-start 之后)的强制存图间隔(分钟)")
    ap.add_argument("--night-start", default="18:30", help="夜间起点 HH:MM")
    ap.add_argument("--night-end", default="06:00", help="夜间终点 HH:MM(回到白天频率)")
    ap.add_argument("--filter-conf", type=float, default=0.5,
                    help="画面里检出置信度 >= 该值的目标就跳过不存(防把真石头当背景)")
    ap.add_argument("--no-filter", action="store_true", help="跳过'有目标就不存'的检查")
    ap.add_argument("--device", default="0", help="过滤模型用的推理设备")
    ap.add_argument("--out", default=str(ROOT / "dataset" / "images" / "train"))
    ap.add_argument("--val-dir", default=str(ROOT / "dataset" / "images" / "valid"))
    ap.add_argument("--val-every", type=int, default=10, help="每 N 张抽 1 张进验证集")
    ap.add_argument("--prefix", default="neg")
    args = ap.parse_args()

    out_dir = Path(args.out)
    val_dir = Path(args.val_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 过滤模型: 画面里已经检出目标就别当背景存(见模块 docstring 最后一条)
    model = None
    if not args.no_filter:
        try:
            from ultralytics import YOLO
            w = ROOT / "runs" / "train" / "yolo11s_v2" / "weights" / "best.pt"
            if w.exists():
                model = YOLO(str(w))
                print(f"[filter] 启用目标过滤: {w}  conf>={args.filter_conf}", flush=True)
            else:
                print(f"[warn] 找不到权重 {w}，本次不启用目标过滤", flush=True)
        except Exception as exc:
            print(f"[warn] 加载过滤模型失败({exc})，本次不启用", flush=True)

    lasts, saved = [None] * len(args.rtsp), [0] * len(args.rtsp)
    last_save_t = [0.0] * len(args.rtsp)      # 每路上次存图时刻，用于"时间兜底"

    max_gap = max(args.force_minutes, args.night_minutes) * 60.0
    cur_night = None                      # 白天/夜间切换时打一条日志，便于确认时间表生效
    t0 = time.time()

    def grab_frame(url):
        """连上 -> 预热几帧 -> 立刻断开，全程只占一路连接。

        86 那台 NVR 的并发子码流上限约 10 路，实况已占 6 路；采集如果再
        常驻 6 路 = 12 路，NVR 直接拒绝，采不到图还挤掉实况。
        采集本来就是每路几分钟一张，抓完就断，不跟实况抢连接。
        """
        cap = open_rtsp(url)
        if cap is None or not cap.isOpened():
            return None
        frame = None
        for _ in range(5):            # 刚连上前几帧常是解码残帧，多读几张挑一张能用的
            ok, f = cap.read()
            if ok and f is not None and f.std() >= 8:
                frame = f
            time.sleep(0.05)
        cap.release()
        return frame

    while min(saved) < args.count:
        for i, url in enumerate(args.rtsp):
            if saved[i] >= args.count:
                continue
            frame = grab_frame(url)
            if frame is None:
                print(f"[cam{i}] 取流失败，下一轮再试", flush=True)
                continue
            # 静止画面永远判定为"没变化"，只靠变化触发会一张都存不出来(之前的 bug)；
            # 所以加一条时间兜底：距上次存图超过当前时段的间隔就强制存一张，
            # 这样才能覆盖早/中/晚的光照变化，而不是存一堆几乎一样的图。
            force_gap, night = force_gap_now(args)
            if cur_night != night:
                cur_night = night
                print(f"[sched] {'夜间' if night else '白天'}模式: 每 {force_gap / 60:.0f} 分钟一张",
                      flush=True)
            forced = (force_gap > 0 and last_save_t[i] > 0
                      and time.time() - last_save_t[i] >= force_gap)
            if not forced and lasts[i] is not None and looks_same(lasts[i], frame):
                continue                      # 画面没变化就不重复存
            if model is not None:
                res = model.predict(frame, imgsz=640, conf=args.filter_conf,
                                    device=args.device, verbose=False)[0]
                if res.boxes is not None and len(res.boxes):
                    print(f"[cam{i}] 画面里有 {len(res.boxes)} 个目标，跳过不存", flush=True)
                    continue
            lasts[i] = frame.copy()
            last_save_t[i] = time.time()
            saved[i] += 1
            stamp = time.strftime("%Y%m%d_%H%M%S")
            to_val = args.val_every > 0 and saved[i] % args.val_every == 0
            d = val_dir if to_val else out_dir
            p = d / f"{args.prefix}cam{i}_{stamp}_{saved[i]:04d}.jpg"
            cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            print(f"[cam{i}] {saved[i]}/{args.count} -> {p.name}"
                  + (" (val)" if to_val else ""), flush=True)
        time.sleep(args.interval)
        # 静止场景下主要靠"时间兜底"出图，实际耗时接近 count * force_gap 而不是 count * interval，
        # 所以超时阈值要把 force_gap 算进去，否则采到一半就被这个守卫踢出去。
        if time.time() - t0 > args.count * max(args.interval, max_gap) * len(args.rtsp) + 3600:
            print("[warn] 超时退出(有源长期连不上)", flush=True)
            break

    print(f"[done] 各路采集: {saved}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
