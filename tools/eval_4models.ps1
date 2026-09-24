# 四个模型在同一测试集(split_v3 test)上的逐类别评估，结果落 runs/val/<name>/per_class.json
#   yolov8s-v1 / yolov8s-v2 / yolov11s-v1 / yolov11s-v2
# conf 用 0.001（和训练期验证一致），不能用实况的 0.25，否则召回被人为压低不好横向比。
$PY = "C:\Users\8\Desktop\rk3588\.venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }

$env:PYTHONIOENCODING = "utf-8"

Set-Location (Join-Path $PSScriptRoot "..")

$jobs = @(
    , @("yolov8s-v1",  "runs/train/yolov8s_v1/weights/best.pt")
    , @("yolov8s-v2",  "runs/train/yolov8s_v2/weights/best.pt")
    , @("yolov11s-v1", "models/yolov11s_v1/best.pt")
    , @("yolov11s-v2", "runs/train/yolo11s_v2/weights/best.pt")
)

foreach ($j in $jobs) {
    $name = $j[0]
    $w = $j[1]
    Write-Output "===== $name ($w) ====="
    & $PY tools/eval_per_class.py `
        --weights $w `
        --data dataset/splits/split_v3.yaml `
        --split test `
        --split-file dataset/splits/split_v3_test.txt `
        --name "${name}_test" `
        --batch 16 --device 0 --workers 2 --conf 0.001
}
Write-Output "===== EVAL DONE ====="
