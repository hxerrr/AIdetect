"""YOLO11 RKNN/ONNX 通用后处理

输入: 模型输出张量, shape (1, 7, 8400) 或 (7, 8400)
      7 = 4(box: xywh) + nc(3 类分数)
      8400 = 80x80 + 40x40 + 20x20

输出: list[(cls_id, conf, x1, y1, x2, y2)]，坐标为输入尺寸(letterbox 后)下的像素值
"""

import cv2
import numpy as np


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0
    y[..., 2] = x[..., 0] + x[..., 2] / 2.0
    y[..., 3] = x[..., 1] + x[..., 3] / 2.0
    return y


def nms_xyxy(boxes, scores, iou_thresh):
    """纯 numpy 类别内 NMS，boxes 为 xyxy。
    注意: 不能用 cv2.dnn.NMSBoxes，它要求 [x,y,w,h]，传 xyxy 会算错 IoU 导致过度抑制。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[ovr <= iou_thresh]
    return keep


def decode_yolo(output, conf_thresh=0.25, iou_thresh=0.45, max_det=300):
    """解析 YOLO11 检测头输出并做 NMS"""
    if output is None:
        return []
    if output.ndim == 3:
        output = output[0]
    # 兼容 (7, 8400) 与 (8400, 7) 两种排布 -> (8400, 7)
    if output.shape[0] < output.shape[-1]:
        output = output.T
    boxes_xywh = output[:, :4]
    scores = output[:, 4:]  # (8400, nc)

    cls_ids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)
    mask = confs > conf_thresh
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    xyxy = xywh2xyxy(boxes_xywh)

    # 按类别分别 NMS，与 Ultralytics 默认行为一致（类别无关 NMS 需设置 agnostic=True）
    keep = []
    for c in np.unique(cls_ids):
        mask_c = cls_ids == c
        boxes_c = xyxy[mask_c]
        confs_c = confs[mask_c]
        ids_c = np.where(mask_c)[0]
        for idx in nms_xyxy(boxes_c, confs_c, iou_thresh):
            keep.append(int(ids_c[idx]))

    if not keep:
        return []
    # 按置信度排序取 top max_det
    order = np.argsort([-confs[i] for i in keep])[:max_det]
    keep = [keep[i] for i in order]

    dets = []
    for i in keep:
        x1, y1, x2, y2 = xyxy[i]
        dets.append((int(cls_ids[i]), float(confs[i]), float(x1), float(y1), float(x2), float(y2)))
    return dets


def letterbox(img, target_size=640, color=(114, 114, 114)):
    """保持长宽比 resize/pad 到目标正方形尺寸"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_right = target_size - new_w - pad_left

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, pad_left, pad_top


def scale_coords(coords, img0_shape, input_size=640, pad_left=0, pad_top=0, scale=1.0):
    """把输入尺寸下的 xyxy 映射回原图"""
    coords = np.array(coords, dtype=np.float32)
    coords[..., [0, 2]] -= pad_left
    coords[..., [1, 3]] -= pad_top
    coords[..., :4] /= scale
    # 裁剪到原图边界
    coords[..., [0, 2]] = coords[..., [0, 2]].clip(0, img0_shape[1])
    coords[..., [1, 3]] = coords[..., [1, 3]].clip(0, img0_shape[0])
    return coords
