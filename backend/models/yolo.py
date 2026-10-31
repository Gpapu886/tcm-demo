# -*- coding: utf-8 -*-
"""YOLO11n 舌象属性检测（ONNX CPU 推理，DEMO 版）

模型：训练自 9 类舌象属性数据集（红舌/紫暗舌/胖大舌/瘦薄舌/红点舌/裂纹舌/齿痕舌/白苔/黄苔）
部署：ultralytics 导出 ONNX -> onnxruntime 推理（无需 GPU，2C4G 云主机可跑）
"""

import os

import cv2
import numpy as np

MODEL_PATH = os.environ.get("YOLO_ONNX", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "best.onnx"))

ATTRS = ["hongshe", "zishe", "pangdashe", "shoushe", "hongdianshe",
         "liewenshe", "chihenshe", "baitaishe", "huangtaishe"]

IMG_SIZE = 640
CONF_THR = 0.15
IOU_THR = 0.5

_session = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"未找到 ONNX 模型: {MODEL_PATH}\n"
                "请先运行 scripts/export_onnx.py（在学校 A800 上）导出后放入 models/best.onnx")
        _session = ort.InferenceSession(
            MODEL_PATH, providers=["CPUExecutionProvider"])
        print(f"[yolo] ONNX 已加载: {MODEL_PATH}")
    return _session


def _letterbox(img: np.ndarray, size: int) -> tuple:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


def detect(image: np.ndarray) -> dict:
    """检测 9 类舌象属性，返回：
    {
      "boxes": [{attr, attr_cn, conf, x1, y1, x2, y2(原图坐标)}, ...],
      "probs": {attr: max_conf, ...}
    }
    """
    session = _get_session()
    input_name = session.get_inputs()[0].name
    img0 = image
    if img0 is None or img0.size == 0:
        return {"boxes": [], "probs": {a: 0.0 for a in ATTRS}}

    inp, r, dx, dy = _letterbox(img0, IMG_SIZE)
    blob = inp[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[None]

    out = session.run(None, {input_name: blob})[0]  # (1, 8400, 4+nc) 或 (1, 4+nc, 8400)
    out = out[0]
    if out.shape[0] != (4 + nc) and out.shape[1] == (4 + nc):
        out = out.T  # (8400, 4+nc) -> (4+nc, 8400)
    if out.shape[0] != (4 + nc):
        raise ValueError(f"ONNX 输出形状异常: {out.shape}")

    nc = len(ATTRS)
    boxes = out[:4].T           # 8400 x 4 (xywh, letterbox 坐标)
    scores = out[4:4 + nc].T    # 8400 x 9

    dets = []
    for i in range(out.shape[1]):
        cls_id = int(np.argmax(scores[i]))
        conf = float(scores[i][cls_id])
        if conf < CONF_THR:
            continue
        cx, cy, w, h = boxes[i]
        x1 = (cx - w / 2 - dx) / r
        y1 = (cy - h / 2 - dy) / r
        x2 = (cx + w / 2 - dx) / r
        y2 = (cy + h / 2 - dy) / r
        dets.append({"attr": ATTRS[cls_id], "conf": round(conf, 3),
                     "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2)})

    # 每属性取最高置信度（demo 简化，不做 NMS 也能展示）
    probs = {a: 0.0 for a in ATTRS}
    for d in dets:
        if d["conf"] > probs[d["attr"]]:
            probs[d["attr"]] = d["conf"]
    for d in dets:
        d["attr_cn"] = {"hongshe": "红舌", "zishe": "紫暗舌", "pangdashe": "胖大舌",
                        "shoushe": "瘦薄舌", "hongdianshe": "红点舌", "liewenshe": "裂纹舌",
                        "chihenshe": "齿痕舌", "baitaishe": "白苔", "huangtaishe": "黄苔"}[d["attr"]]

    return {"boxes": sorted(dets, key=lambda x: -x["conf"])[:6], "probs": probs}
