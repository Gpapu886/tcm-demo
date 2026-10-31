import json
import numpy as np

# 加载知识桥：舌象属性 → 证候支持度
with open("models/bridge_syndrome_by_tongue.json", "r", encoding="utf-8") as f:
    BRIDGE = json.load(f)

# 中文 → 英文 属性名映射
CN2EN = {
    "红舌": "hongshe", "紫舌": "zishe", "胖大舌": "pangdashe",
    "瘦舌": "shoushe", "红点舌": "hongdianshe", "裂纹舌": "liewenshe",
    "齿痕舌": "chihenshe", "白苔": "baitaishe", "黄苔": "huangtaishe"
}

class UCMArbiter:
    def __init__(self, K_T=5.0, K_V=5.0, reject_thresh=0.85):
        self.K_T = K_T    # 文本证据强度
        self.K_V = K_V    # 视觉证据强度
        self.reject_thresh = reject_thresh

    def _visual_to_syndrome(self, visual_attrs):
        # 视觉属性 → 证候分布（用知识桥加权）
        syndrome_score = {}
        for attr_cn, conf in visual_attrs.items():
            attr_en = CN2EN.get(attr_cn)
            if not attr_en or attr_en not in BRIDGE:
                continue
            top_probs = BRIDGE[attr_en].get("top_probs", {})
            for syn, prob in top_probs.items():
                syndrome_score[syn] = syndrome_score.get(syn, 0.0) + conf * prob
        # 归一化
        total = sum(syndrome_score.values())
        if total <= 0:
            return {}
        return {k: v/total for k, v in syndrome_score.items()}

    def arbitrate(self, text_top3, visual_attrs):
        """
        text_top3: [{"syndrome": "xxx", "conf": 0.8}, ...]
        visual_attrs: {"红舌": 0.7, "白苔": 0.9, ...}
        """
        # 1. 文本分布
        text_dist = {item["syndrome"]: item["conf"] for item in text_top3}
        # 2. 视觉证候分布
        visual_dist = self._visual_to_syndrome(visual_attrs)

        # 3. Dirichlet证据融合
        all_syndromes = sorted(set(list(text_dist.keys()) + list(visual_dist.keys())))
        alpha = {}
        for syn in all_syndromes:
            a = 1.0  # 先验
            a += self.K_T * text_dist.get(syn, 0.0)
            a += self.K_V * visual_dist.get(syn, 0.0)
            alpha[syn] = a

        S = sum(alpha.values())
        fused_dist = {k: v/S for k, v in alpha.items()}
        final_syn = max(fused_dist, key=fused_dist.get)
        final_conf = fused_dist[final_syn]

        # 4. 计算不确定性u_alea
        u_alea = sum(q * (1 - q) for q in fused_dist.values())

        # 5. 分级建议
        if u_alea < self.reject_thresh:
            suggestion = "高置信自动输出"
            level = "high"
        else:
            suggestion = "建议人工复核"
            level = "low"

        # Top3排序
        top3 = sorted(fused_dist.items(), key=lambda x: -x[1])[:3]
        top3_result = [{"syndrome": k, "conf": round(v, 4)} for k, v in top3]

        return {
            "final_syndrome": final_syn,
            "final_confidence": round(final_conf, 4),
            "u_alea": round(u_alea, 4),
            "suggestion": suggestion,
            "confidence_level": level,
            "top3": top3_result
        }
