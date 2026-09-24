# 批量用 CPU 跑运动目标检测（GPU 被训练占用时使用）
# 用法: powershell -File tools/motion_batch.ps1
$ErrorActionPreference = "Continue"
$py = "c:\Users\8\Desktop\rk3588\.venv\Scripts\python.exe"
$w = "runs/detect/runs/train/yolo11s_v2-4/weights/best.pt"
$in = "E:\Project\AIdetect\video\input"
$out = "E:\Project\AIdetect\video\output"

Write-Output "=== 批量运动目标检测 (CPU) ==="
Write-Output "weights: $w"

Get-ChildItem "$in\*.mp4" | ForEach-Object {
    $dst = Join-Path $out ($_.BaseName + "_motion.mp4")
    Write-Output "--- $($_.Name) -> $dst"
    & $py tools/motion_tracker.py --weights $w --source $_.FullName --device cpu --output $dst
}

Write-Output "--- images -> $out\images"
& $py tools/motion_tracker.py --weights $w --source "$in\images" --device cpu --output "$out\images"

Write-Output "=== ALL DONE ==="
