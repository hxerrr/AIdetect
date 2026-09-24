# yolov8s_v3 训练：DST 增强集(不含负样本)，对照实验用
#
# 数据集: dataset/splits/dst_no_neg.yaml
#   train = 41330 张增强图(排除 2750 张 neg_* 负样本)
#   valid/test = 1046/716，与 dst_neg.yaml 一致
# 目的: 和 yolov8s_v2(同架构同参数，唯独多了负样本) 对照，量化负样本的贡献。
#
# 参数与 yolov8s_v2 完全一致，唯一变量是"有没有负样本"
$PY = "C:\Users\8\Desktop\rk3588\.venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }

$env:PYTHONIOENCODING = "utf-8"      # 否则日志里的中文会变乱码

Set-Location (Join-Path $PSScriptRoot "..")
& $PY train.py `
  --weights yolov8s.pt `
  --data dataset/splits/dst_no_neg.yaml `
  --name yolov8s_v3 `
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
