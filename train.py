"""AIdetect YOLO11s 训练脚本

用法示例:
    # 冒烟测试
    python train.py --epochs 1 --name smoke
    # 正式训练（v2-4 实际使用的配置，已固化为默认值）
    python train.py --name yolo11s_v2-5

说明:
  - 布尔参数用 --no-xxx 关闭（如 --no-amp / --no-cos-lr）。
    早期版本用 type=bool 是个坑: 传 --amp False 会被解析成 True。
  - 默认不启用 disk 缓存。cache=disk 会把约 25GB 的 .npy 写进 dataset/images/，
    且改完数据集后必须手工删除 .npy 才会生效，否则训练读的是旧缓存。
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="yolo11s.pt")
    p.add_argument(
        "--init",
        type=str,
        default="",
        help="当 --weights 是 .yaml(改结构如 P2 头)时，用该权重做部分初始化。"
             "只载入 key 与 shape 都匹配的参数，新增分支随机初始化。",
    )
    p.add_argument("--data", type=str, default="data.yaml")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="-1 表示自动探测最大 batch")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--optimizer", type=str, default="AdamW")
    p.add_argument("--lr0", type=float, default=0.002)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--close-mosaic", type=int, default=20)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)
    # 注意: copy_paste 仅对带 polygon 分割标注的数据生效，纯 bbox 数据集会被静默跳过
    p.add_argument("--copy-paste", type=float, default=0.0)
    p.add_argument("--auto-augment", type=str, default="randaugment")
    p.add_argument("--erasing", type=float, default=0.4)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--project", type=str, default="runs/train")
    p.add_argument("--name", type=str, default="yolo11s_aidetect")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache", type=str, default="", help="空=不缓存, ram, disk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    from ultralytics import YOLO

    a = parse_args()

    # 固化工作目录 + project 绝对路径，避免结果落到 runs/detect/runs/train 这类嵌套目录
    os.chdir(ROOT)
    project = Path(a.project)
    if not project.is_absolute():
        project = ROOT / project
    print(f"[train] cwd={os.getcwd()} save_dir={project / a.name}")

    model = YOLO(a.weights)

    # 结构改动(如加 P2 头)后从预训练权重部分初始化：只载入 key 与 shape 都匹配的参数，
    # 新增/通道变化的层保持随机初始化，避免从头训练。
    if a.init:
        # 用 ultralytics 自己的加载器，避免 torch>=2.6 的 weights_only=True
        # 无法反序列化 DetectionModel 的问题（直接 torch.load 会抛 UnpicklingError）
        src_sd = YOLO(a.init).model.state_dict()
        dst_sd = model.model.state_dict()
        matched = {k: v for k, v in src_sd.items() if k in dst_sd and dst_sd[k].shape == v.shape}
        dst_sd.update(matched)
        model.model.load_state_dict(dst_sd)
        print(f"[init] 从 {a.init} 载入 {len(matched)}/{len(src_sd)} 个参数张量"
              f"（目标模型共 {len(dst_sd)} 个）")

    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        optimizer=a.optimizer,
        lr0=a.lr0,
        lrf=a.lrf,
        cos_lr=a.cos_lr,
        warmup_epochs=a.warmup_epochs,
        patience=a.patience,
        close_mosaic=a.close_mosaic,
        mosaic=a.mosaic,
        mixup=a.mixup,
        copy_paste=a.copy_paste,
        auto_augment=a.auto_augment,
        erasing=a.erasing,
        scale=a.scale,
        project=str(project),
        name=a.name,
        resume=a.resume,
        cache=a.cache or False,
        seed=a.seed,
        amp=a.amp,
        val=True,
        plots=True,
        save=True,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
