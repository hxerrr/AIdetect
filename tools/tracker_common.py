"""AIdetect 运动目标跟踪 / 时序滤波 —— PC 本地验证与 RK3588 板端共用

只依赖 numpy（绘图函数额外依赖 cv2），不依赖 ultralytics，便于整体拷贝到板端。

相比初版实现的改进（对应复查发现的问题）:
  1. 恒速外推预测: 匹配前用 EMA 速度把上一帧框外推 dt 秒再算 IoU，解决落石这类
     高速小目标相邻帧位移过大、IoU 掉到阈值以下而断轨的问题。
  2. 全局运动补偿: 用当前帧所有匹配轨迹位移的**中位数**估计相机抖动/云台转动并扣除，
     避免"整幅画面一起动"造成的集体误报。
  3. 速度按 dt 归一化为"归一化位移/秒"(相对图像对角线)，帧率波动时阈值不漂移。
  4. 滞回确认: 连续 min_motion_frames 帧超阈值才判定为运动，抑制检测框抖动造成的误报。
  5. 事件精简为 new / moving 两类，不再为每一次匹配发事件（旧实现每 2s 刷一条静态事件）。
  6. 固定物抑制(static_after>0，默认关闭): 静止超过该秒数的目标判定为"固定物"(背景石头等)，
     不再进事件表，画面上画灰色细框。轨迹断 ID 后重新检出会按位置认出来(fixture_ttl 秒内)，
     避免同一个石头反复告警。开启时 new 事件也改为"确认运动后才发"。

用法:
    from tracker_common import MotionTracker, draw_tracks

    tracker = MotionTracker(motion_threshold=0.10)
    tracks = tracker.update(dets, img_h, img_w)   # dets: [(cls, conf, x1, y1, x2, y2), ...]
    events = tracker.flush_events()
"""

import math
import time
from collections import deque

import numpy as np

# 默认 3 类口径（当前数据集）
NAMES = {0: "landslide", 1: "collapse", 2: "rockfall"}

# 按"类别名"而非"类别 id"映射颜色：不同代模型的类别编号并不一致
#   v0 模型: 0=fallen tree, 1=landslide, 2=road collapse, 3=stone
#   v2 模型: 0=landslide,   1=collapse,   2=rockfall
# 只有按名字取色，横向对比时同类目标才会是同一种颜色。
NAME2COLOR = {
    "landslide": (255, 0, 0),       # 蓝
    "collapse": (0, 0, 255),        # 红
    "road collapse": (0, 0, 255),   # 红
    "rockfall": (0, 255, 0),        # 绿
    "stone": (0, 255, 0),           # 绿
    "fallen tree": (0, 255, 255),   # 黄
}
DEFAULT_COLOR = (128, 128, 128)


def color_for(label):
    return NAME2COLOR.get(str(label).lower(), DEFAULT_COLOR)


class Track:
    """单条时序轨迹"""

    def __init__(self, tid, cls, conf, box, img_h, img_w, history=10, names=None):
        self.id = tid
        self.cls = int(cls)
        # 用模型自带的 names，避免不同代模型类别编号不一致导致标签错配
        self.label = (names or NAMES).get(self.cls, str(self.cls))
        self.conf = float(conf)
        self.box = [float(v) for v in box]
        self.img_h = img_h
        self.img_w = img_w
        self.diag = math.hypot(img_w, img_h) or 1.0
        self.centers = deque(maxlen=history)   # 最近 N 帧中心点
        self.speeds = deque(maxlen=history)    # 最近 N 帧速度(归一化位移/秒)
        self.hit = 1                           # 累计匹配帧数
        self.miss = 0                          # 连续未匹配帧数
        self.age = 1
        self.motion_run = 0                    # 连续超阈值帧数(滞回计数)
        self.last_event = {}                   # kind -> 上次事件时间，按类型分别冷却
        self.announced = False                 # 是否已发过 new 事件
        self.is_active = False
        self.born = time.time()                # 轨迹创建时刻，用于算"静止了多久"
        self.last_move_time = None             # 最近一次**确认**运动的时刻
        self.is_fixture = False                # 静止超时 -> 固定物(背景石头等)，不再进事件表
        self.vx = 0.0                          # 像素/秒
        self.vy = 0.0
        self.centers.append(self.center)
        # 轨迹出生时的中心点：用于算"净位移"。
        # 静止目标的框抖动会让逐帧速度看起来一直在动(截图案例 speed=0.1265 的桌上石头)，
        # 但中心点始终在原地打转，净位移接近 0 —— 靠这个能把假 Moving 剔掉。
        self.start_center = self.center

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def predicted_box(self, dt):
        """恒速外推 dt 秒后的框，用于对抗高速目标的帧间大位移"""
        dx, dy = self.vx * dt, self.vy * dt
        x1, y1, x2, y2 = self.box
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def update(self, box, conf, speed, vx, vy):
        self.box = [float(v) for v in box]
        self.conf = float(conf)
        self.vx, self.vy = float(vx), float(vy)
        self.speeds.append(float(speed))
        self.centers.append(self.center)
        self.hit += 1
        self.miss = 0
        self.age += 1

    def predict(self):
        """本帧未匹配，仅推进寿命"""
        self.miss += 1
        self.age += 1

    def mean_speed(self):
        return float(np.mean(self.speeds)) if self.speeds else 0.0

    def net_disp_ratio(self):
        """从轨迹出生到现在的净位移 ÷ 图像对角线。

        用来区分「真的挪了位置」和「原地抖动」：静止目标的框一直在颤，
        逐帧速度看起来非零，但中心点始终没离开原点。
        """
        sx, sy = self.start_center
        cx, cy = self.center
        return math.hypot(cx - sx, cy - sy) / (self.diag or 1.0)


class MotionTracker:
    """IoU 贪婪匹配 + 恒速外推 + 全局运动补偿 + 滞回运动判定

    motion_threshold 单位: 归一化位移/秒（位移 ÷ 图像对角线 ÷ dt）。
    参考值: 0.10 ≈ 1080p 下 220 px/s；调大更保守，调小更灵敏。
    """

    def __init__(
        self,
        iou_threshold=0.3,
        max_miss=5,
        min_hits=3,
        motion_threshold=0.10,
        event_cooldown=2.0,
        min_motion_frames=3,
        history=10,
        default_fps=25.0,
        compensate=True,
        static_after=0.0,
        fixture_ttl=60.0,
        names=None,
        min_event_hits=None,
        min_disp_ratio=0.0,
    ):
        self.names = names
        self.iou_threshold = iou_threshold
        self.max_miss = max_miss
        self.min_hits = min_hits
        self.motion_threshold = motion_threshold
        self.event_cooldown = event_cooldown
        self.min_motion_frames = min_motion_frames
        self.history = history
        self.default_fps = default_fps
        self.compensate = compensate
        # static_after>0 时启用固定物抑制：静止超过该秒数的目标不再进事件表。
        # 默认 0 = 关闭，保持与板端 board_infer 的历史行为一致。
        self.static_after = static_after
        self.fixture_ttl = fixture_ttl
        # 事件确认帧数：轨迹要连续命中这么多帧才允许报事件(低于此只画框不报)。
        # 默认 None = 沿用 min_hits，保持板端/历史行为；给 8 能挡掉反光这类只闪几帧的误检。
        self.min_event_hits = min_hits if min_event_hits is None else int(min_event_hits)
        # 净位移门槛(占图像对角线比例)：中心点从轨迹出生起必须真的挪动这么远才算 Moving。
        # 0 = 关闭(历史行为)；0.03 约等于 640x360 下 22px。
        self.min_disp_ratio = float(min_disp_ratio)

        self.tracks = []
        self.fixtures = []                     # 已知固定物 {cls, box, until}，防止断 ID 后重复告警
        self.events = []
        self.next_id = 1
        self.global_motion = (0.0, 0.0)   # 本帧估计出的相机整体位移(px)，便于调试/上屏
        self._last_ts = None

    # ---------------- 基础工具 ----------------
    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / (union + 1e-6)

    def _now_dt(self, timestamp):
        now = timestamp if timestamp is not None else time.time()
        if self._last_ts is None:
            dt = 1.0 / max(self.default_fps, 1e-3)
        else:
            dt = now - self._last_ts
            # 首帧/卡顿/取流间隔异常时兜底，避免 dt 极端值把速度算爆
            if not (1e-3 <= dt <= 1.0):
                dt = 1.0 / max(self.default_fps, 1e-3)
        self._last_ts = now
        return now, dt

    # ---------------- 主流程 ----------------
    def update(self, detections, img_h, img_w, timestamp=None):
        """detections: list[(cls, conf, x1, y1, x2, y2)]，坐标为原图像素"""
        now, dt = self._now_dt(timestamp)
        diag = math.hypot(img_w, img_h) or 1.0

        dets = [(int(d[0]), float(d[1]), [float(d[2]), float(d[3]), float(d[4]), float(d[5])]) for d in detections]
        n_d, n_t = len(dets), len(self.tracks)

        # 1) 用外推框做 IoU 贪婪匹配（同类之间），先匹配 IoU 高的
        pred_boxes = [t.predicted_box(dt) for t in self.tracks]
        pairs = []
        for i in range(n_t):
            ti = self.tracks[i]
            for j in range(n_d):
                if ti.cls != dets[j][0]:
                    continue
                iou = self._iou(pred_boxes[i], dets[j][2])
                if iou >= self.iou_threshold:
                    pairs.append((iou, i, j))
        pairs.sort(reverse=True)

        matched_t, matched_d = set(), set()
        matched = []  # (track, box, conf, dx, dy)
        for _, i, j in pairs:
            if i in matched_t or j in matched_d:
                continue
            matched_t.add(i)
            matched_d.add(j)
            t = self.tracks[i]
            cls, conf, box = dets[j]
            px, py = t.center
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            matched.append((t, box, conf, cx - px, cy - py))

        # 2) 全局运动补偿：多数目标静止时，位移中位数即相机自身抖动/云台转动
        gdx = gdy = 0.0
        if self.compensate and len(matched) >= 3:
            arr = np.array([[m[3], m[4]] for m in matched], dtype=np.float32)
            gdx, gdy = (float(v) for v in np.median(arr, axis=0))
        self.global_motion = (gdx, gdy)

        # 3) 更新已匹配轨迹
        for t, box, conf, dx, dy in matched:
            dx -= gdx
            dy -= gdy
            vx, vy = dx / dt, dy / dt
            speed = math.hypot(dx, dy) / (diag * dt)
            t.update(box, conf, speed, vx, vy)
            t.motion_run = t.motion_run + 1 if speed > self.motion_threshold else 0
            # 确认运动(连续 min_motion_frames 帧)才刷新，避免框抖动把固定物"救活"
            if t.motion_run >= self.min_motion_frames:
                t.last_move_time = now
                t.is_fixture = False

        # 4) 未匹配的检测 -> 新轨迹
        for j in range(n_d):
            if j in matched_d:
                continue
            cls, conf, box = dets[j]
            trk = Track(self.next_id, cls, conf, box, img_h, img_w,
                        history=self.history, names=self.names)
            trk.born = now
            # 落在已知固定物上(轨迹断 ID 后重新检出) -> 直接继承固定物身份，不再重复告警
            if self._match_fixture(cls, box, now):
                trk.is_fixture = True
            self.tracks.append(trk)
            self.next_id += 1

        # 5) 未匹配的轨迹 -> 标记丢失
        for i in range(n_t):
            if i not in matched_t:
                self.tracks[i].predict()
                self.tracks[i].motion_run = 0

        # 6) 删除长期丢失的轨迹
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

        # 7) 状态更新与事件判定
        for t in self.tracks:
            t.is_active = (t.hit >= self.min_hits) and (t.miss == 0)
            if not t.is_active:
                continue
            # 静止超过 static_after 秒 -> 固定物(背景里的石头等)，不进事件表
            if self.static_after > 0 and not t.is_fixture:
                ref = t.last_move_time if t.last_move_time is not None else t.born
                if now - ref > self.static_after:
                    t.is_fixture = True
            if t.is_fixture:
                self._remember_fixture(t, now)
                continue
            # 事件确认帧数：闪几帧就消失的目标(地面反光、灯光光斑)不报
            if t.hit < self.min_event_hits:
                continue
            # 滞回 + 均值双确认：反光/抖动造成的瞬时速度尖峰能把 motion_run 顶上阈值，
            # 但把均值拉不起来(截图案例: 瞬时连续超阈、均值仅 0.0671，画面实为 Static)。
            # 两个条件同时满足才算运动，压制"闪烁式"误报。
            moving = (t.motion_run >= self.min_motion_frames
                      and t.mean_speed() > self.motion_threshold * 0.8
                      and t.net_disp_ratio() >= self.min_disp_ratio)
            # 抑制开启时 new 要等目标真正动起来才发：新出现的目标是不是背景石头
            # 得等满 static_after 才知道，先发就会在判定为固定物前先刷一条出来
            if moving or self.static_after <= 0:
                if not t.announced:
                    t.announced = True
                    self._emit(t, "new", now)
            if moving:
                self._emit(t, "moving", now)

        return self.tracks

    # ---------------- 固定物(长期静止目标)抑制 ----------------
    def _match_fixture(self, cls, box, now):
        """新检出是否落在已知固定物上：轨迹断 ID 后重新出现时用来抑制重复告警"""
        for f in self.fixtures:
            if f["until"] > now and f["cls"] == cls and self._iou(f["box"], box) >= 0.5:
                return True
        return False

    def _remember_fixture(self, t, now):
        """登记/续期固定物，ttl 内再出现能直接认出来"""
        self.fixtures = [f for f in self.fixtures if f["until"] > now]
        for f in self.fixtures:
            if f["cls"] == t.cls and self._iou(f["box"], t.box) >= 0.5:
                f["box"] = list(t.box)
                f["until"] = now + self.fixture_ttl
                return
        self.fixtures.append({"cls": t.cls, "box": list(t.box), "until": now + self.fixture_ttl})

    def _emit(self, t, kind, now):
        # 按事件类型分别冷却：否则 "new" 会吃掉 "moving" 的额度，
        # 导致目标首次被判定为运动时要等一个冷却周期才发得出事件。
        last = t.last_event.get(kind)
        if last is not None and now - last < self.event_cooldown:
            return
        t.last_event[kind] = now
        self.events.append(
            {
                "time": now,
                "track_id": t.id,
                "cls": t.cls,
                "label": t.label,
                "kind": kind,
                "speed": round(t.mean_speed(), 5),
                "conf": round(t.conf, 3),
                "x1": float(t.box[0]),
                "y1": float(t.box[1]),
                "x2": float(t.box[2]),
                "y2": float(t.box[3]),
            }
        )

    def flush_events(self):
        e = self.events
        self.events = []
        return e

    def active_tracks(self):
        return [t for t in self.tracks if t.is_active]


def draw_tracks(frame, tracks, events=None, motion_threshold=0.10, show_trail=True,
                include_inactive=False, min_disp_ratio=0.0):
    """在帧上绘制轨迹框 / 中心轨迹 / 事件提示

    include_inactive: 单张静态图没有时序概念，轨迹永远到不了 min_hits，
                      默认会被跳过导致图上没有任何框。此时需置 True 直接画出检测结果。
    """
    import cv2

    for t in tracks:
        if not t.is_active and not include_inactive:
            continue
        # 配色：固定物=青色(灰框在土石背景里根本看不清)、运动中=红色(需要告警的)、其余=类别色
        is_fixture = getattr(t, "is_fixture", False)
        speed = t.mean_speed()
        # 与事件判定同一套口径：净位移不够的一律画成 Static，避免画面显示 Moving 但不告警的矛盾
        disp = t.net_disp_ratio() if hasattr(t, "net_disp_ratio") else 1.0
        moving = speed > motion_threshold and disp >= min_disp_ratio
        if is_fixture:
            color = (255, 255, 0)
        elif moving:
            color = (0, 0, 255)
        else:
            color = color_for(t.label)
        x1, y1, x2, y2 = map(int, t.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        status = "Fixed" if is_fixture else ("Moving" if moving else "Static")
        # 文字加同色底条 + 黑字：裸文字叠在画面上经常和背景融为一体
        label = f"{t.label}-{t.id} {t.conf:.2f} {status}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + bl + 6)
        cv2.rectangle(frame, (x1, ty - th - bl - 3), (x1 + tw + 8, ty + bl + 2), color, -1)
        cv2.putText(frame, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
        if show_trail:
            centers = list(t.centers)
            for i in range(1, len(centers)):
                cv2.line(frame, tuple(map(int, centers[i - 1])), tuple(map(int, centers[i])), color, 1)

    if events:
        y0 = 30
        for e in events[-5:]:
            cv2.putText(
                frame,
                f"{e['label']}-{e['track_id']} {e['kind']} speed={e['speed']:.4f}",
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            y0 += 25
    return frame
