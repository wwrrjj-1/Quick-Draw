"""
Quick Draw - FastAPI 后端

提供模型推理 API，供前端调用。
"""

import os
import sys
from pathlib import Path
import base64
import io

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# 添加项目根目录
sys.path.append(str(Path(__file__).parent))
from src.config import BEST_MODEL_PATH, DEVICE, CATEGORIES
from src.model import get_model

# ==========================================
# 模型加载
# ==========================================

model = None
categories = None
device = None

def load_model():
    """加载模型"""
    global model, categories, device
    
    # 强制使用 CPU（部署时不依赖 CUDA）
    device = torch.device("cpu")
    print(f"[INFO] Device: {device}")
    
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}")
    
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    num_classes = checkpoint.get('num_classes', 345)
    categories = checkpoint.get('categories', CATEGORIES[:num_classes])
    model_type = checkpoint.get('model_type', 'base')
    
    model = get_model(num_classes=num_classes, model_type=model_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Model loaded: {model_type}, classes: {num_classes}")

# ==========================================
# FastAPI 应用
# ==========================================

app = FastAPI(title="Quick Draw API")

class PredictRequest(BaseModel):
    image: str  # Base64 编码的图像数据

class PredictResponse(BaseModel):
    success: bool
    predictions: list

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    return FileResponse("static/index.html")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """预测 API"""
    global model, categories, device
    
    try:
        # 解码 Base64 图像
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        original_img = Image.open(io.BytesIO(image_bytes))
        
        print(f"[DEBUG] Original image size: {original_img.size}, mode: {original_img.mode}")
        
        # 转换为灰度
        img = original_img.convert('L')
        
        # 调整大小为 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        
        print(f"[DEBUG] Before invert: min={img_array.min():.0f}, max={img_array.max():.0f}")
        
        # Quick Draw 数据格式：白色笔画(255)在黑色背景(0)上
        # Canvas 格式：深色笔画(~50)在白色背景(255)上
        # 需要反转：255 - pixel_value
        img_array = 255.0 - img_array
        
        # 对比度增强：将笔画强化到 255
        # 训练数据笔画是纯白(255)，用户输入经过反转后约 200
        # 使用阈值二值化或对比度拉伸
        if img_array.max() > 0:
            # 方法1：对比度拉伸（将最大值拉到 255）
            img_array = img_array * (255.0 / img_array.max())
            
            # 方法2：阈值强化（可选，让笔画更清晰）
            # threshold = 30
            # img_array = np.where(img_array > threshold, 255, 0)
        
        print(f"[DEBUG] After enhance: min={img_array.min():.0f}, max={img_array.max():.0f}, mean={img_array.mean():.1f}")
        
        # 保存调试图像
        debug_img = Image.fromarray(img_array.astype(np.uint8), mode='L')
        debug_img.save('debug_input.png')
        
        # 归一化到 [0, 1]
        img_array = img_array / 255.0
        
        # 调试：打印有效像素范围
        print(f"[DEBUG] Final: min={img_array.min():.3f}, max={img_array.max():.3f}, "
              f"mean={img_array.mean():.3f}, stroke_pixels={(img_array > 0.1).sum()}")
        
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        # Top-5 结果
        top5_probs, top5_indices = torch.topk(probs, 5)
        
        predictions = []
        for prob, idx in zip(top5_probs, top5_indices):
            idx = idx.item()
            category = categories[idx] if idx < len(categories) else f"class_{idx}"
            predictions.append({
                "category": category,
                "confidence": round(float(prob) * 100, 1)
            })
        
        return PredictResponse(success=True, predictions=predictions)
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return PredictResponse(success=False, predictions=[])

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    print("=" * 50)
    print("Quick Draw - 启动服务")
    print("访问: http://localhost:7860")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=7860)
