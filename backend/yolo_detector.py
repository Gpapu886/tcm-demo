import cv2
import numpy as np
import onnxruntime as ort

# 9类属性，和训练时顺序完全一致
CLASS_NAMES = [
    "hongshe", "zishe", "pangdashe", "shoushe",
    "hongdianshe", "liewenshe", "chihenshe",
    "baitaishe", "huangtaishe"
]
# 中文名字映射
CLASS_CN = {
    "hongshe": "红舌", "zishe": "紫舌", "pangdashe": "胖大舌",
    "shoushe": "瘦舌", "hongdianshe": "红点舌", "liewenshe": "裂纹舌",
    "chihenshe": "齿痕舌", "baitaishe": "白苔", "huangtaishe": "黄苔"
}

class YOLODetector:
    def __init__(self, onnx_path="models/best.onnx", imgsz=640, conf_thres=0.15):
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img_bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.original_h, self.original_w = img.shape[:2]
        
        # 等比缩放+灰边填充，强制最终尺寸严格=640x640
        r = min(self.imgsz / self.original_w, self.imgsz / self.original_h)
        new_w, new_h = int(round(self.original_w * r)), int(round(self.original_h * r))
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 计算补边量
        pad_w = self.imgsz - new_w
        pad_h = self.imgsz - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        # 强制兜底：保证最终尺寸严格是640x640，杜绝1像素误差
        img_padded = cv2.resize(img_padded, (self.imgsz, self.imgsz))
        
        # 记录缩放系数，用于坐标还原
        self.ratio = r
        self.pad_left = pad_left
        self.pad_top = pad_top

        # BGR转RGB，归一化，转CHW格式
        img_input = img_padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img_input, axis=0)

    def predict(self, img_bytes):
        input_data = self._preprocess(img_bytes)
        outputs = self.session.run(None, {self.input_name: input_data})[0]
        
        # 兼容两种输出维度
        if outputs.shape[1] == 8400:
            outputs = outputs.transpose(0, 2, 1)
        outputs = outputs[0]

        attr_probs = {name: 0.0 for name in CLASS_NAMES}
        boxes = []

        for i in range(outputs.shape[1]):
            cls_scores = outputs[4:, i]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if conf < self.conf_thres:
                continue
            cls_name = CLASS_NAMES[cls_id]
            if conf > attr_probs[cls_name]:
                attr_probs[cls_name] = conf
            # 坐标还原到原图
            cx, cy, w, h = outputs[0, i], outputs[1, i], outputs[2, i], outputs[3, i]
            x1 = int((cx - w/2 - self.pad_left) / self.ratio)
            y1 = int((cy - h/2 - self.pad_top) / self.ratio)
            x2 = int((cx + w/2 - self.pad_left) / self.ratio)
            y2 = int((cy + h/2 - self.pad_top) / self.ratio)
            boxes.append({
                "x1": max(0, x1), "y1": max(0, y1),
                "x2": min(self.original_w, x2), "y2": min(self.original_h, y2),
                "cls": cls_name, "conf": round(conf, 3)
            })

        # 转中文结果返回
        attr_result = {CLASS_CN[k]: round(v, 3) for k, v in attr_probs.items() if v > 0}
        return attr_result, boxes
