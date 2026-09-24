# yolov11s_v3 训练：最好的架构(yolo11s) x 最好的数据(DST 增强集 + 2750 张负样本)
#
# 背景: 同一份 split_v3 原数据上，yolov8s-v1(0.554) 和 yolov11s-v2(0.548) 几乎打平，
#       说明架构不是瓶颈；换成 DST 后 yolov8s-v2 直接 0.554 -> 0.808。
#       所以把 yolo11s 也放到 DST+负样本上训一版，看能不能再超过 0.808。
#
# 数据集: dataset/splits/dst_neg.yaml
#   train = 41330 张增强图 + 2750 张纯背景负样本 = 44080 张
#   valid/test 保持原样(1046/716)，mAP 能和 yolov8s-v2 直接对比
#
# 参数与 yolov8s_v2 完全一致(唯一变量是架构)，只改 --weights 和 --name
$PY = "C:\Users\8\Desktop\rk3588\.venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }

$env:PYTHONIOENCODING = "utf-8"      # 否则日志里的中文会变乱码

Set-Location (Join-Path $PSScriptRoot "..")
& $PY train.py `
  --weights yolo11s.pt `
  --data dataset/splits/dst_neg.yaml `
  --name yolov11s_v3 `
  --epochs 100 `
  --imgsz 640 `
  --batch 16 `
  --device 0 `
  --workers 4 `
  --mixup 0.1 `
  --close-mosaic 15 `
  --patience 30 `
  --optimizer AdamW `
  --lr0 0.002 `
  --lrf 0.01 `
  --cos-lr `
  --warmup-epochs 3 `
  --auto-augment randaugment `
  --erasing 0.4 `
  --scale 0.5 `
  --seed 0 `
  --amp
