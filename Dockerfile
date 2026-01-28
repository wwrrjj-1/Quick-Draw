# Quick Draw - Docker 部署配置
# 使用 Python slim 镜像 + PyTorch CPU 版本

FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用 PyTorch CPU 版本）
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .
COPY static/ ./static/
COPY src/ ./src/
COPY models/ ./models/

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "app.py"]
