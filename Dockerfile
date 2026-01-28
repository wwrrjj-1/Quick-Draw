FROM python:3.10-slim
LABEL "language"="python"
LABEL "framework"="fastapi"

WORKDIR /app

# 安装系统依赖 - 只安装必要的库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
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

EXPOSE 7860

CMD ["python", "app.py"]
