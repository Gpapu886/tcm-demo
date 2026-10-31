# -*- coding: utf-8 -*-
"""U-CMEA 仲裁模块（DEMO 版）

对应论文中的 U-CMEA（Uncertainty-aware Cross-Modal Evidence Arbitration）：
  1) 文本支路给出证候 top3；
  2) 视觉支路给出 9 类舌象属性置信度；
  3) 二者经 K_T / K_V 加权融合为证候分布，计算 u_alea（aleatoric uncertainty）；
  4) 高不确定性样本建议转人工复核（选择性预测）。

正式参数（与论文实验一致）：K_T = 5.0，K_V = 5.0。
"""

import json
import os

# ---- 正式参数 ----
K_T = 5.0
K_V = 5.0

# ---- 舌象属性（9 类，与 YOLO 检测头一致） ----
ATTRS = [
    "hongshe", "zishe", "pangdashe", "shoushe", "hongdianshe",
    "liewenshe", "chihenshe", "baitaishe", "huangtaishe",
]
ATTR_CN = {
    "hongshe": "红舌", "zishe": "紫暗舌", "pangdashe": "胖大舌", "shoushe": "瘦薄舌",
    "hongdianshe": "红点舌", "liewenshe": "裂纹舌", "chihenshe": "齿痕舌",
    "baitaishe": "白苔", "huangtaishe": "黄苔",
}

# ---- 轻量知识桥（内置简化版，可在 data/ 放完整 bridge_syndrome_by_tongue.json 覆盖） ----
# 属性 -> {证候: 提升权重}，来自"舌象属性-证候共现"先验
LIGHT_BRIDGE = {
    "hongshe":    {"阴虚火旺证": 1.8, "热毒蕴结证": 1.6, "湿热内蕴证": 1.5, "肝火炽盛证": 1.5},
    "zishe":      {"血瘀痰凝证": 2.0, "气滞血瘀证": 1.8, "寒凝血瘀证": 1.4},
    "pangdashe":  {"脾虚湿盛证": 1.9, "痰湿内阻证": 1.7, "气虚证": 1.4, "阳虚证": 1.3},
    "shoushe":    {"阴虚证": 1.8, "气血两虚证": 1.6, "津亏证": 1.4},
    "hongdianshe": {"热毒蕴结证": 1.6, "血热证": 1.5, "阴虚火旺证": 1.3},
    "liewenshe":  {"阴虚证": 1.7, "津液亏虚证": 1.6, "血虚证": 1.3},
    "chihenshe":  {"脾虚湿盛证": 1.8, "气虚证": 1.5, "痰湿内阻证": 1.4},
    "baitaishe":  {"寒湿困脾证": 1.6, "痰湿内阻证": 1.4, "脾虚湿盛证": 1.3},
    "huangtaishe": {"湿热内蕴证": 1.9, "痰热壅肺证": 1.7, "肝火炽盛证": 1.4},
}

_BRIDGE = None  # 完整桥（若存在则优先）


def load_bridge(path: str) -> dict:
    """从完整知识桥文件加载：{attr: {syndrome: lift, ...}, ...}"""
    global _BRIDGE
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _BRIDGE = json.load(f)
        print(f"[cmea] 已加载完整知识桥: {path}")
    else:
        _BRIDGE = None


def bridge() -> dict:
    return _BRIDGE if _BRIDGE else LIGHT_BRIDGE


def fuse(text_top3: list, visual_probs: dict) -> dict:
    """U-CMEA 融合：返回 {syndrome: prob}（归一化分布）

    text_top3   : [{"syndrome": str, "prob": float}, ...]（生成模型输出，prob 可为 0/1 排序）
    visual_probs: {attr: conf}（YOLO 9 类检测置信度，缺省为 0）
    """
    br = bridge()
    # 候选空间 = 文本 top3 ∪ 视觉桥命中的证候
    space = {}
    for item in text_top3:
        s = item.get("syndrome", "").strip()
        if s:
            space[s] = space.get(s, 0.0) + max(float(item.get("prob", 0.0)), 1e-6)
    ts = text_top3[0]["syndrome"] if text_top3 else ""

    for attr, conf in visual_probs.items():
        if conf <= 0 or attr not in br:
            continue
        for syn, lift in br[attr].items():
            space[syn] = space.get(syn, 0.0) + 1e-6  # 占位，先保证空间存在

    if not space:
        return {}

    # 加权得分：先验 1 + 文本权重 K_T + 视觉证据 K_V * lift
    score = {}
    for syn in space:
        a = 1.0
        if ts and syn == ts:
            a += K_T
        for attr, conf in visual_probs.items():
            if conf <= 0 or attr not in br:
                continue
            a += K_V * conf * br[attr].get(syn, 0.0)
        score[syn] = a

    total = sum(score.values())
    return {syn: sc / total for syn, sc in score.items()}


def uncertainty(dist: dict) -> dict:
    """计算 u_alea（aleatoric）与置信度，给出选择性建议"""
    if not dist:
        return {"u_alea": 1.0, "conf": 0.0, "top1": "", "advice": "reject", "advice_cn": "无可靠证据，建议转人工"}
    total = sum(dist.values())
    p = {k: v / total for k, v in dist.items()}
    top1 = max(p, key=p.get)
    u_alea = sum(q * (1 - q) for q in p.values())
    conf = p[top1]
    # 选择性阈值（论文口径简化）：u_alea 高于 0.90 或置信度过低 → 建议转人工
    if u_alea > 0.90 or conf < 0.25:
        advice, advice_cn = "refer", "把握不足，建议转人工复核"
    else:
        advice, advice_cn = "adopt", "可采纳"
    return {"u_alea": round(u_alea, 4), "conf": round(conf, 4),
            "top1": top1, "advice": advice, "advice_cn": advice_cn}
