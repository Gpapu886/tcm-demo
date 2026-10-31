# -*- coding: utf-8 -*-
"""导出 YOLO11n ONNX（在学校 A800 / 服务器上运行一次即可）

用法：
    conda activate py310
    pip install ultralytics
    python export_onnx.py /public/home/sszkhoc/tongue_raw/runs/detect/train-8/weights/best.pt

产物：models/best.onnx（约 10MB，CPU 即可推理）
"""
import sys
from pathlib import Path

from ultralytics import YOLO


def main():
    if len(sys.argv) < 2:
        print("用法: python export_onnx.py <best.pt 路径>")
        sys.exit(1)
    pt = sys.argv[1]
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(pt)
    model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
    # ultralytics 默认输出到 pt 同目录，移动过来
    src = Path(pt).with_suffix(".onnx")
    dst = out_dir / "best.onnx"
    if src.exists():
        import shutil
        shutil.move(str(src), str(dst))
    print(f"完成: {dst}")
    # 打印 9 类 names 供核对
    names = model.names
    if isinstance(names, dict):
        print("names:", [names[i] for i in range(len(names))])


if __name__ == "__main__":
    main()
