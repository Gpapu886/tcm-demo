import requests
import json

# 这里用DeepSeek的API，最便宜，效果也够；后面你可以换成豆包/通义千问
API_URL = "https://api.deepseek.com/chat/completions"
# 后面你自己去DeepSeek开放平台注册，把key填进来
API_KEY = "你的API_KEY"

SYSTEM_PROMPT = """
你是一名专业的中医助理医师，请根据用户输入的症状描述和病历信息，给出最可能的3个中医证候诊断，按置信度从高到低排序。
输出严格为JSON格式，不要多余文字，格式如下：
{"top3": [{"syndrome": "脾虚湿盛证", "conf": 0.85}, {"syndrome": "痰湿内阻证", "conf": 0.1}, {"syndrome": "肝郁脾虚证", "conf": 0.05}]}
置信度加和为1，证候名称用标准中医术语，带"证"字结尾。
"""

def diagnose_text(symptom_text):
    # 演示模式：没有API key时，返回模拟结果，方便你先跑通界面
    if not API_KEY or API_KEY == "你的API_KEY":
        return [
            {"syndrome": "脾虚湿盛证", "conf": 0.72},
            {"syndrome": "痰湿内阻证", "conf": 0.18},
            {"syndrome": "肝郁脾虚证", "conf": 0.10}
        ]

    # 真实调用API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "deepseek-chat",
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": symptom_text}
        ]
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=10)
        result = resp.json()
        content = result["choices"][0]["message"]["content"].strip()
        # 解析JSON
        if content.startswith("```json"):
            content = content[7:-3].strip()
        parsed = json.loads(content)
        return parsed["top3"]
    except Exception as e:
        print(f"API调用失败: {e}")
        # 失败返回模拟结果
        return [
            {"syndrome": "脾虚湿盛证", "conf": 0.72},
            {"syndrome": "痰湿内阻证", "conf": 0.18},
            {"syndrome": "肝郁脾虚证", "conf": 0.10}
        ]
