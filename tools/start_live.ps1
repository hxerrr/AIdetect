# 多路监控(海康子码流)一键启动
#
# 通道0-5: 192.168.110.86  NVR，实测 1/2/3/4/5/6/7/8 通道都存在
#          (101..801 主码流 2560x1440、102..802 子码流 640x360)
#          注意: 与 capture_negatives.py 的默认六路保持一致，改这里也要改那边。
#          (96 球机已撤下，不再接)
# 九宫格最多 9 路，当前占 6 路；通道7/8(702/802) 或那五台独立机
#   210/214/215/216/217(只有通道1: .../Channels/102) 想接就往下面加 --rtsp。
# 重新扫通道: python tools/probe_channels.py --host 192.168.110.86 --password "Yunfan@2025"
#
# 误检抑制参数(按需调):
#   --conf 0.35           低于 0.35 的一律不报。人被认成 collapse/rockfall 的置信度
#                         在 0.38~0.48，0.5 会把这些事件帧全挡掉，降到 0.35 让它们
#                         能触发事件并存进训练集(confirm-hits/位移/面积等约束仍在)
#   --confirm-hits 8      连续命中 8 帧才报事件，挡只闪几帧的反光/光斑
#   --min-disp-ratio 0.03 净位移不够 3% 对角线不算 Moving，挡静止石头被框抖动误报
#   --edge-margin 0.06    贴边且面积 >=10% 画面的大框丢弃(镜头前失焦遮挡物)
#
# 直接用 venv 里的解释器绝对路径：系统 PATH 上的 python 是应用商店别名，
# 调起来静默退出且没有 torch/ultralytics，必须指到实体解释器。
$PY = "C:\Users\8\Desktop\rk3588\.venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }

$env:PYTHONIOENCODING = "utf-8"      # 否则日志里的中文会变乱码

Set-Location (Join-Path $PSScriptRoot "..")
& $PY tools/rtsp_web_detect.py `
  --rtsp "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/702" `
  --rtsp "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/402" `
  --rtsp "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/202" `
  --rtsp "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/302" `
  --rtsp "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/502" `
  --rtsp "rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/602" `
  --weights runs/train/yolo11s_v2/weights/best.pt `
  --conf 0.35 `
  --confirm-hits 8 `
  --min-disp-ratio 0.03 `
  --edge-margin 0.06 `
  --edge-area 0.10 `
  --imgsz 640 --half --device 0 --port 5000
