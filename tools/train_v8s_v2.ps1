# yolov8s_v2 训练：DST 增强集 + 2750 张负样本(negatives_aug)
#
# 数据集: dataset/splits/dst_neg.yaml
#   train = 41330 张增强图 + 2750 张纯背景负样本 = 44080 张
#   valid/test 保持原样(1046/716)，这样 mAP 能和 yolo11s_v2 直接对比
#   负样本标签全为空(0 字节)，专门压"人/反光/遮挡被认成落石"这类误检
#
# 泛化/鲁棒性相关参数(相对默认值的微调):
#   --mixup 0.1        两图线性混合，逼模型别只靠单点纹理下判断，抗误检
#   --close-mosaic 15  最后 15 轮关掉 mosaic，让模型在真实尺度上收敛
#                      (100 轮的 15%，默认 20 是按 150 轮设的)
#   --auto-augment randaugment + --erasing 0.4
#                      随机遮挡，模拟监控画面里的局部遮挡/逆光
#   --scale 0.5        尺度抖动，适配远近不同的镜头
#   --patience 30      100 轮里 30 轮不涨就早停，省时间
#   --no-cache         不开 disk 缓存(会写 ~25GB .npy 到数据集目录，且改数据后易踩旧缓存)
#
# 和 yolo11s_v2 保持一致的部分: AdamW + cos_lr + lr0 0.002 + warmup 3 + imgsz 640
$PY = "C:\Users\8\Desktop\rk3588\.venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }

$env:PYTHONIOENCODING = "utf-8"      # 否则日志里的中文会变乱码

Set-Location (Join-Path $PSScriptRoot "..")
& $PY train.py `
  --weights yolov8s.pt `
  --data dataset/splits/dst_neg.yaml `
  --name yolov8s_v2 `
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
