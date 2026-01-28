# Quick Draw - 部署指南

## 📦 项目结构

```
quick_draw/
├── app.py              # FastAPI 后端
├── static/index.html   # 前端页面
├── src/                # 源代码模块
├── models/             # 训练好的模型
├── requirements.txt    # Python 依赖（CPU版）
├── Dockerfile          # Docker 配置
└── .dockerignore       # Docker 忽略文件
```

## 🚀 部署方式

### 方式一：直接运行

```bash
# 安装依赖（CPU 版本，约 200MB）
pip install -r requirements.txt

# 启动服务
python app.py
```

访问 http://localhost:7860

### 方式二：Docker 部署

```bash
# 构建镜像
docker build -t quick-draw .

# 运行容器
docker run -p 7860:7860 quick-draw
```

### 方式三：云平台部署

#### Hugging Face Spaces
1. 创建新 Space，选择 Docker
2. 上传项目文件
3. 自动构建并部署

#### Railway / Render
1. 连接 Git 仓库
2. 设置启动命令：`python app.py`
3. 暴露端口：7860

## ⚙️ 环境要求

- Python 3.10+
- 内存：512MB+
- 存储：~500MB（含模型）

## 📝 注意事项

1. **模型文件**：确保 `models/quick_draw_best.pth` 存在
2. **端口配置**：默认 7860，可在 `app.py` 中修改
3. **CPU 模式**：已配置为纯 CPU 推理，无需 GPU
