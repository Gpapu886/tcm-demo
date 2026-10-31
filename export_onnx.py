from ultralytics import YOLO

# 加载你训练的9类YOLO模型
model = YOLO("models/best.pt")

# 导出成ONNX格式，imgsz和你训练时一致
model.export(format="onnx", imgsz=640, opset=12)

print("导出完成！models文件夹里会出现 best.onnx")
