# -*- coding: utf-8 -*-
"""文本支路：病历文本 -> 证候 top3（大模型 API，OpenAI 兼容格式）

支持 DeepSeek / 豆包 / 通义等任何 OpenAI 兼容接口：
    export LLM_API_KEY=sk-xxx
    export LLM_BASE_URL=https://api.deepseek.com/v1   # 可换
    export LLM_MODEL=deepseek-chat                     # 可换
"""

import json
import os

from openai import OpenAI

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

_SYS = (
    "你是一名资深中医师。请根据患者病历（主诉、现病史、既往史、四诊信息），"
    "辨证给出最可能的证候。只输出 JSON，不要任何解释。"
    "格式：{\"syndromes\":[{\"syndrome\":\"证候名\",\"prob\":0.8,\"reason\":\"一句话依据\"}, ...]}，"
    "按可能性降序输出 3 个，prob 为 0-1 且合计约等于 1。"
)


def available() -> bool:
    return bool(API_KEY)


def _client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def syndrome_top3(text: str) -> list:
    """返回 [{"syndrome": str, "prob": float, "reason": str}, ...]，最多 3 个"""
    client = _client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYS},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    content = resp.choices[0].message.content
    content = content.strip()
    # 兼容 markdown ```json ... ``` 包裹
    if content.startswith("```"):
        content = content.strip("`").lstrip("json").strip()
    data = json.loads(content)
    out = []
    for it in data.get("syndromes", []):
        out.append({
            "syndrome": str(it.get("syndrome", "")).strip(),
            "prob": float(it.get("prob", 0.0)),
            "reason": str(it.get("reason", "")).strip(),
        })
    return out[:3]
