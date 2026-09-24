"""汇总各模型在视频上的运动事件计数，便于横向对比。

motion_tracker.py 的 done 行形如:
    done: 595 frames, avg inference=8.0ms, events: new=33 moving=9

本脚本把 "--- <模型> / <视频>.mp4" 与随后的 done 行配对，输出按视频对齐的对照表。

用法:
    python tools/summarize_video_events.py \
        --logs yolov8s-v1=video/output/v8s1_videos.log \
               yolov11s-v1=video/output/v11s1_videos.log \
               yolov11s-v2=video/output/v2_videos.log
"""
import argparse
import re
from pathlib import Path

RE_HEAD = re.compile(r"^---\s+\S+\s+/\s+(.+?)\.(mp4|avi|mkv|mov)\s*$", re.I)
RE_DONE = re.compile(r"done:\s+(\d+)\s+frames.*?new=(\d+)\s+moving=(\d+)")

W_NAME = 24
W_CELL = 22


def parse(path):
    """返回 {视频名: (帧数, new事件, moving事件)}"""
    rows, cur = {}, None
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    for line in txt.splitlines():
        line = line.strip()
        m = RE_HEAD.match(line)
        if m:
            cur = m.group(1)
            continue
        m = RE_DONE.search(line)
        if m and cur:
            rows[cur] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            cur = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="格式: 模型名=日志文件")
    a = ap.parse_args()

    specs = []
    for s in a.logs:
        name, _, p = s.partition("=")
        specs.append((name.strip(), parse(p.strip())))

    order = []
    for _, r in specs:
        for v in r:
            if v not in order:
                order.append(v)
    order.sort()

    head = "视频".ljust(W_NAME) + "".join(n.rjust(W_CELL) for n, _ in specs)
    print("\n=== 视频运动事件对比 (new=新增目标, moving=持续运动) ===")
    print(head)
    print("-" * len(head.encode("gbk", "ignore")) if False else "-" * (W_NAME + W_CELL * len(specs)))

    for v in order:
        cells = []
        for _, r in specs:
            if v in r:
                nf, new, mv = r[v]
                cells.append(("%d新/%d动" % (new, mv)).rjust(W_CELL))
            else:
                cells.append("--".rjust(W_CELL))
        print(v.ljust(W_NAME) + "".join(cells))

    print("-" * (W_NAME + W_CELL * len(specs)))
    tot = []
    for _, r in specs:
        tot.append((sum(x[1] for x in r.values()), sum(x[2] for x in r.values())))
    print("合计".ljust(W_NAME) + "".join(
        ("%d新/%d动" % t).rjust(W_CELL) for t in tot))
    print()


if __name__ == "__main__":
    main()
