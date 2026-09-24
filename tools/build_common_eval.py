"""构建"两个模型都没见过"的公共评测集，用于公平对比不同代模型

背景: 两代模型用的数据集划分不同，直接比各自 valid 的 mAP 不公平
  - yolo11s_v1(旧划分)   : 训练 17842 张，旧 test(716) 对它干净
  - yolov8s_v1(split_v3) : 训练 15753 张，split_v3 test(1333) 对它干净

互评会双向泄漏: split_v3 valid 里约 91% 的图曾是 yolo11s_v1 的训练图；
                旧 valid 里约 80% 的图是 yolov8s_v1 的训练图。

做法: 取"旧 test"中【不在 split_v3 train 列表里】的图。
      这些图 yolo11s_v1 没训过(属旧 test)，yolov8s_v1 也没训过(不在 split_v3 train)。

用法:
    python tools/build_common_eval.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
OUT = ROOT / "dataset" / "splits"

old_test_dir = ROOT / "dataset" / "images" / "test"
v3_train = OUT / "split_v3_train.txt"

v3_stems = set()
if v3_train.exists():
    for line in v3_train.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            v3_stems.add(Path(line).stem)

keep = [
    p for p in sorted(old_test_dir.rglob("*"))
    if p.is_file() and p.suffix.lower() in EXTS and p.stem not in v3_stems
]

print(f"split_v3 train 图片数  : {len(v3_stems)}")
print(f"对两模型都干净的公共图 : {len(keep)}")

OUT.mkdir(parents=True, exist_ok=True)
lst = OUT / "common_eval.txt"
lst.write_text("\n".join(str(p.resolve()) for p in keep) + "\n", encoding="utf-8")

yaml_p = OUT / "common_eval.yaml"
yaml_p.write_text(
    "# 两代模型都没见过的公共评测集(旧 test 中剔除 split_v3 train 部分)\n"
    f"path: {(ROOT / 'dataset').resolve().as_posix()}\n"
    f"train: {lst.resolve().as_posix()}\n"
    f"val: {lst.resolve().as_posix()}\n"
    "nc: 3\n"
    "names:\n  0: landslide\n  1: collapse\n  2: rockfall\n",
    encoding="utf-8",
)
print(f"[write] {lst}")
print(f"[write] {yaml_p}")
