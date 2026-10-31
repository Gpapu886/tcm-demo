# 中医多模态辅助诊疗系统 DEMO

基于 **U-CMEA 跨模态证据融合**的中医辅助诊疗演示系统：融合文本病历辨证与舌象视觉检测，输出证候诊断 + 不确定性量化 + 分级复核建议。

## 项目定位
面向临床辅助诊疗场景，通过实验**证伪**了「端到端注入视觉可提升强文本基线」的直觉，提出「视觉不做预测、做证据仲裁」的范式：文本有把握就信文本，文本摇摆时视觉出来投票，低置信样本建议人工复核——符合医疗场景「少答错优于多答对」的原则。

## 系统架构
前端（HTML + JavaScript）
↓  RESTful API
后端（FastAPI）
├─ 文本支路：大模型 API → 证候 Top3 置信度
├─ 视觉支路：YOLO11n（ONNX）→ 9 类舌象属性检测
└─ 融合仲裁：U-CMEA Dirichlet 证据融合
→ 最终证候 + 不确定度 u_alea + 分级建议


## 技术栈
- **后端**：FastAPI + Uvicorn，前后端分离，RESTful 接口
- **视觉模型**：YOLO11n → ONNX 轻量化部署，CPU 单图推理 < 200ms
- **文本辨证**：可接入主流大模型 API（DeepSeek / 豆包等）
- **核心算法**：U-CMEA 跨模态证据融合 + 选择性预测 + Conformal 校准

## 核心功能
1. 舌象检测可视化：上传舌图，实时绘制 9 类舌象属性检测框与置信度
2. 文本证候辨证：输入症状/病历，输出 Top3 中医证候
3. U-CMEA 融合仲裁：双支路证据加权融合，计算证据不确定度 u_alea
4. 分级复核建议：高置信自动输出，低置信转人工复核

## 实验结果（553 对配对测试集）
| 指标 | 数值 | 说明 |
|---|---|---|
| 舌象检测 mAP50 | 0.446 | YOLO11n，9 类属性，训练集 7345 张 |
| 文本基线准确率 | 0.881 | 文本支路单模 |
| 端到端注入 | 0.9331 → 0.9331 | 证伪：预测层视觉融合无增量 |
| **U-CMEA 选择性预测** | **0.982** | 采纳前 20% 样本，较基线 +5.4pt |
| Conformal 覆盖 | 93.2% | 90% 目标覆盖下达标（候选集 9 类） |

## 数据构成
| 数据集 | 来源 | 规模 |
|---|---|---|
| TCM-Tongue | 北京工商大学 + 中国中医科学院西苑医院 + 北大人民医院（arXiv 2025） | 5594 训练图 |
| TonguExpert | 复旦现代人类学重点实验室 + 中科院上海营养与健康所（Phenomics 2025） | 人工筛选金标准 1747 张 |

## 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python run.py

# 3. 浏览器打开
http://localhost:8000


tcm-demo/
├── run.py                    # 启动入口
├── requirements.txt
├── backend/
│   ├── main.py               # FastAPI 主应用
│   ├── yolo_detector.py      # 视觉支路：YOLO ONNX 推理
│   ├── text_diagnosis.py     # 文本支路：大模型 API 辨证
│   └── cmea_arbitration.py   # U-CMEA 融合仲裁
├── frontend/
│   └── index.html            # 单页交互前端
└── models/
    └── bridge_syndrome_by_tongue.json  # 属性-证候知识桥



## 说明

本项目为硕士毕业设计《基于多模态融合大模型的中医辅助诊疗系统研究》的工程化演示，仅供技术展示，不构成临床诊疗建议。

**② `requirements.txt`（放 tcm-demo 根目录）**
```txt
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
onnxruntime>=1.16.0
opencv-python>=4.8.0
numpy>=1.24.0
requests>=2.31.0
```