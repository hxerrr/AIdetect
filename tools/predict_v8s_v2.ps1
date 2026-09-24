# 用训练好的 yolov8s_v2 跑 video/input 下的视频和图片
#   视频 -> video/output/yolov8s-v2/<原名>.mp4
#   图片 -> video/output/yolov8s-v2/images/<原名>.jpg
# 同时导出检测框 txt(带置信度)，方便和别的结果对比。
#
# conf 0.25: 和实况一个量级；评估才用 0.001 那种极限阈值。
$PY = "C:\Users\8\Desktop\rk3588\.venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }

$env:PYTHONIOENCODING = "utf-8"

Set-Location (Join-Path $PSScriptRoot "..")
& $PY tools/predict_v8s_v2.py `
  --weights runs/train/yolov8s_v2/weights/best.pt `
  --name yolov8s-v2 `
  --device 0
