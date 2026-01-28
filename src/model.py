"""
Quick Draw CNN 模型定义

定义用于涂鸦分类的卷积神经网络模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QuickDrawCNN(nn.Module):
    """
    Quick Draw 涂鸦分类 CNN 模型
    
    架构：
        输入: 1×28×28
        → Conv2d(32) → BatchNorm → ReLU → MaxPool
        → Conv2d(64) → BatchNorm → ReLU → MaxPool
        → Conv2d(128) → BatchNorm → ReLU → MaxPool
        → Flatten → FC(512) → Dropout → FC(num_classes)
    """
    
    def __init__(self, num_classes: int = 345, dropout: float = 0.5):
        """
        初始化模型
        
        Args:
            num_classes: 类别数量
            dropout: Dropout 比率
        """
        super(QuickDrawCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # 卷积层 1: 1 -> 32 通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 卷积层 2: 32 -> 64 通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 卷积层 3: 64 -> 128 通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        # 28 -> 14 -> 7 -> 3 (经过 3 次池化)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch, 1, 28, 28)
        
        Returns:
            输出张量，形状 (batch, num_classes)
        """
        # 卷积块 1: (batch, 1, 28, 28) -> (batch, 32, 14, 14)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积块 2: (batch, 32, 14, 14) -> (batch, 64, 7, 7)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积块 3: (batch, 64, 7, 7) -> (batch, 128, 3, 3)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平: (batch, 128, 3, 3) -> (batch, 128*3*3)
        x = x.view(-1, 128 * 3 * 3)
        
        # 全连接层
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征向量（用于可视化）
        
        Args:
            x: 输入张量
        
        Returns:
            特征向量，形状 (batch, 512)
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        return x


class QuickDrawCNNLarge(nn.Module):
    """
    更大的 Quick Draw CNN 模型（用于更高精度）
    
    相比基础模型增加了一层卷积和更多通道。
    """
    
    def __init__(self, num_classes: int = 345, dropout: float = 0.5):
        super(QuickDrawCNNLarge, self).__init__()
        
        self.num_classes = num_classes
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, 1, 28, 28) -> (batch, 64, 14, 14)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # (batch, 64, 14, 14) -> (batch, 128, 7, 7)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # (batch, 128, 7, 7) -> (batch, 256, 3, 3)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # (batch, 256, 3, 3) -> (batch, 512, 1, 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(-1, 512)
        
        # 全连接
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class QuickDrawResNet(nn.Module):
    """
    使用 ResNet18 预训练模型的 Quick Draw 分类器
    
    关键改进：
    - 使用 ImageNet 预训练权重，迁移学习强大特征
    - 将灰度图扩展为 3 通道 RGB
    - 上采样 28x28 到 64x64 适应 ResNet 输入
    """
    
    def __init__(self, num_classes: int = 345, dropout: float = 0.3, pretrained: bool = True):
        super(QuickDrawResNet, self).__init__()
        
        self.num_classes = num_classes
        
        # 加载预训练 ResNet18
        from torchvision import models
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.resnet = models.resnet18(weights=weights)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # 修改第一层卷积以接受更小的输入
        # 原始: conv1(7x7, stride=2) 适合 224x224
        # 修改: conv1(3x3, stride=1) 适合 64x64
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除原始的 maxpool（保留更多空间信息）
        self.resnet.maxpool = nn.Identity()
        
        # 替换最后的全连接层
        in_features = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
        # 上采样层：28x28 -> 64x64
        self.upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch, 1, 28, 28)
        
        Returns:
            输出张量，形状 (batch, num_classes)
        """
        # 上采样: (batch, 1, 28, 28) -> (batch, 1, 64, 64)
        x = self.upsample(x)
        
        # 灰度转 RGB: (batch, 1, 64, 64) -> (batch, 3, 64, 64)
        x = x.repeat(1, 3, 1, 1)
        
        # ResNet 前向传播
        x = self.resnet(x)
        
        return x


def get_model(num_classes: int = 345, 
              model_type: str = "base",
              dropout: float = 0.5,
              pretrained: bool = True) -> nn.Module:
    """
    获取模型实例
    
    Args:
        num_classes: 类别数量
        model_type: 模型类型，"base"、"large" 或 "resnet"
        dropout: Dropout 比率
        pretrained: 是否使用预训练权重（仅对 resnet 有效）
    
    Returns:
        模型实例
    """
    if model_type == "resnet":
        return QuickDrawResNet(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    elif model_type == "large":
        return QuickDrawCNNLarge(num_classes=num_classes, dropout=dropout)
    else:
        return QuickDrawCNN(num_classes=num_classes, dropout=dropout)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch 模型
    
    Returns:
        (可训练参数数, 总参数数)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    """测试模型"""
    # 创建模型
    model = QuickDrawCNN(num_classes=345)
    print(f"基础模型结构:\n{model}\n")
    
    # 统计参数
    trainable, total = count_parameters(model)
    print(f"可训练参数: {trainable:,}")
    print(f"总参数: {total:,}")
    print(f"模型大小: {total * 4 / 1024 / 1024:.2f} MB\n")
    
    # 测试前向传播
    batch = torch.randn(4, 1, 28, 28)
    output = model(batch)
    print(f"输入形状: {batch.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试特征提取
    features = model.get_features(batch)
    print(f"特征形状: {features.shape}")
    
    # 测试大模型
    print("\n" + "=" * 40)
    model_large = QuickDrawCNNLarge(num_classes=345)
    trainable_large, total_large = count_parameters(model_large)
    print(f"大模型可训练参数: {trainable_large:,}")
    print(f"大模型大小: {total_large * 4 / 1024 / 1024:.2f} MB")
