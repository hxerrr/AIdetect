"""训练排队器：等当前 YOLO11s 跑完，用同一套超参自动启动 YOLOv8s（A/B 对比）

用法:
    python tools/train_queue.py --weights yolov8s.pt --name yolov8s_v2
    python tools/train_queue.py --dry-run            # 只打印将要执行的命令，不真跑

结束判定（满足任一即认为上一轮已结束）:
    1) --watch 指向的 results.csv 最后一行 epoch >= --epochs（正常跑满）
    2) results.csv 超过 --idle 分钟没更新（覆盖早停 / 异常退出）
随后再确认没有仍在运行的训练进程，并留出 --cool-down 秒让 GPU 显存释放，然后启动下一轮。

超参固定为 yolo11s_v2-4 的 args.yaml 配置，保证两轮只差模型结构，对比才有效。
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 与 runs/detect/runs/train/yolo11s_v2-4/args.yaml 对齐的超参
SAME_ARGS = [
    "--epochs", "150",
    "--imgsz", "640",
    "--batch", "16",
    "--device", "0",
    "--workers", "4",
    "--optimizer", "AdamW",
    "--lr0", "0.002",
    "--lrf", "0.01",
    "--cos-lr",
    "--warmup-epochs", "3.0",
    "--patience", "50",
    "--close-mosaic", "20",
    "--mosaic", "1.0",
    "--mixup", "0.0",
    "--copy-paste", "0.0",
    "--auto-augment", "randaugment",
    "--erasing", "0.0",
    "--scale", "0.5",
    "--seed", "0",
    "--amp",
    "--cache", "disk",
    "--project", "runs/train",
]


def last_epoch(csv_path: Path):
    try:
        lines = [l for l in csv_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (FileNotFoundError, OSError):
        return None
    if len(lines) < 2:
        return None
    try:
        return int(float(lines[-1].split(",")[0]))
    except (ValueError, IndexError):
        return None


def training_running(keyword: str):
    """是否还有命令行里带 keyword 的 python 训练进程（best effort，失败时返回 False）"""
    if not keyword:
        return False
    ps = [
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*{keyword}*' }} | "
        "Measure-Object | Select-Object -ExpandProperty Count",
    ]
    try:
        out = subprocess.run(ps, capture_output=True, text=True, timeout=60)
        return int((out.stdout or "0").strip().splitlines()[-1] or 0) > 0
    except Exception:  # noqa: BLE001
        return False


def wait_finish(csv_path: Path, epochs: int, idle: int, poll: int, keyword: str, wait_process: int):
    print(f"[watch] {csv_path}", flush=True)
    while True:
        ep = last_epoch(csv_path)
        mtime = csv_path.stat().st_mtime if csv_path.exists() else time.time()
        idle_min = (time.time() - mtime) / 60.0
        if ep is not None:
            print(f"[watch] epoch={ep}/{epochs}  idle={idle_min:.1f}min", flush=True)
        if ep is not None and ep >= epochs:
            print(f"[watch] 已跑满 {epochs} 轮", flush=True)
            break
        if idle_min >= idle:
            print(f"[watch] 超过 {idle} 分钟无更新，判定训练已结束", flush=True)
            break
        time.sleep(poll)

    for i in range(wait_process):
        if not training_running(keyword):
            print("[watch] 训练进程已退出", flush=True)
            return
        print(f"[watch] 训练进程仍在运行，等待中 ({i + 1}/{wait_process})", flush=True)
        time.sleep(20)
    print("[warn] 等待进程退出超时，仍将尝试启动下一轮（若 GPU 被占用会启动失败）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", default="runs/detect/runs/train/yolo11s_v2-4/results.csv")
    ap.add_argument("--keyword", default="yolo11s_v2-4", help="用于判断训练进程是否还在跑的命令行关键字")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--idle", type=int, default=6, help="results.csv 多少分钟没更新就认为结束")
    ap.add_argument("--poll", type=int, default=60, help="轮询间隔(秒)")
    ap.add_argument("--wait-process", type=int, default=30, help="最多等多少个 20 秒让进程退出")
    ap.add_argument("--cool-down", type=int, default=60, help="结束后额外等待的秒数，等 GPU 显存释放")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--name", default="yolov8s_v2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    wait_finish(Path(a.watch), a.epochs, a.idle, a.poll, a.keyword, a.wait_process)

    cmd = [sys.executable, str(ROOT / "train.py"), "--weights", a.weights, "--name", a.name] + SAME_ARGS
    print("[run] " + " ".join(cmd), flush=True)
    if a.dry_run:
        return

    print(f"[run] {a.cool_down}s 后启动，等待 GPU 显存释放...", flush=True)
    time.sleep(a.cool_down)
    ret = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"[run] 训练结束，exitcode={ret}", flush=True)


if __name__ == "__main__":
    main()
