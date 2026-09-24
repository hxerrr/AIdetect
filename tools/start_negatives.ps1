# 负样本(背景图)后台采集 —— 采够后回训才能压住"人/反光被认成落石"这类误检
#
# 默认六路: 96 球机 + 86 NVR 的通道 2/3/4/5/6
# --force-minutes 15  : 每 15 分钟强制存一张(画面没变化也存)。
#                       只靠"画面变化"触发的话，静止场景一张都存不出来(老 bug)。
# --count 150         : 每路 150 张，15min 一张约 37 小时收工(约 96 张/天/路)。
# --no-filter        : 不做"画面里有目标就跳过"的过滤，一律保存。
#                      当前场景没有任何真实地质目标，所有检出都是误检(人/反光/遮挡)，
#                      这些帧正是训练最需要的难例负样本，一张都不能扔。
#                      以后场景里出现真目标时，改回 --filter-conf 0.75 恢复过滤。
$PY = "C:\Users\8\Desktop\rk3588\.venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }

$env:PYTHONIOENCODING = "utf-8"

Set-Location (Join-Path $PSScriptRoot "..")
& $PY tools/capture_negatives.py `
  --count 150 `
  --interval 20 `
  --force-minutes 15 `
  --no-filter `
  --device 0
