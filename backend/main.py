from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import traceback

# 导入三个模块
from .yolo_detector import YOLODetector
from .text_diagnosis import diagnose_text
from .cmea_arbitration import UCMArbiter

app = FastAPI(title="中医多模态辅助诊疗DEMO")

# 全局初始化模型（启动时加载一次）
try:
    detector = YOLODetector()
    arbiter = UCMArbiter(K_T=5.0, K_V=5.0)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    traceback.print_exc()
    detector = None
    arbiter = None

# 挂载前端页面
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    symptom: str = Form(...)
):
    try:
        # 打印接收到的参数，方便排查问题
        print(f"收到请求：图片={image.filename}, 症状长度={len(symptom)}")
        
        if not detector or not arbiter:
            return {"code": 500, "msg": "模型未加载成功，请检查后端日志"}

        # 1. 视觉支路：舌象检测
        img_bytes = await image.read()
        visual_attrs, boxes = detector.predict(img_bytes)

        # 2. 文本支路：证候辨证
        text_top3 = diagnose_text(symptom)

        # 3. U-CMEA融合仲裁
        arbit_result = arbiter.arbitrate(text_top3, visual_attrs)

        # 4. 统一返回格式
        return {
            "code": 200,
            "data": {
                "visual": {
                    "attributes": visual_attrs,
                    "boxes": boxes
                },
                "text_top3": text_top3,
                "arbitration": arbit_result
            }
        }
    except Exception as e:
        # 出错也返回格式化信息，前端不会炸
        print(f"❌ 接口报错: {e}")
        traceback.print_exc()
        return {"code": 500, "msg": f"分析失败：{str(e)}"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
