# Quick Draw 涂鸦识别项目

基于 Google Quick Draw 数据集的深度学习图像分类项目。

## 数据集信息

- **来源**: Google Quick Draw Dataset
- **类别数**: 345 类
- **每类样本**: 10,000 张（抽样）
- **图片尺寸**: 28×28 灰度图
- **总数据量**: ~2.7 GB

## 环境要求

- Python 3.10+
- CUDA 11.8+（GPU 训练）
- 显存 >= 4GB

## 快速开始

```bash
# 1. 激活虚拟环境
conda activate dl-gpu

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据（约 2.7GB）
python src/download_data.py

# 4. 训练模型
python src/train.py

# 5. 评估模型
python src/evaluate.py

# 6. 启动交互演示
python app.py
```

## 项目结构

```
quick_draw/
├── data/           # 数据目录（.npy 文件）
├── models/         # 保存的模型权重
├── src/
│   ├── config.py       # 配置参数
│   ├── download_data.py    # 数据下载脚本
│   ├── dataset.py      # PyTorch Dataset
│   ├── model.py        # CNN 模型
│   ├── train.py        # 训练脚本
│   └── evaluate.py     # 评估脚本
├── app.py          # Gradio 交互界面
├── requirements.txt
└── README.md
```

## 模型性能

训练完成后会在此处更新模型性能指标。
