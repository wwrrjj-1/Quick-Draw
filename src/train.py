"""
Quick Draw 训练脚本（优化版）

支持：
- ResNet18 预训练模型
- 混合精度训练（AMP）
- Label Smoothing
- 数据增强
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    CATEGORIES, DATA_DIR, MODEL_DIR, BEST_MODEL_PATH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE,
    SAMPLES_PER_CLASS, NUM_CLASSES
)
from src.dataset import get_dataloaders
from src.model import get_model, count_parameters


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, 
                    scaler=None, use_amp=True):
    """
    训练一个 epoch（支持混合精度）
    
    Returns:
        (平均损失, 准确率)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [训练]", leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 混合精度前向传播
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通训练
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, desc="验证"):
    """
    验证/测试
    
    Returns:
        (平均损失, 准确率)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"[{desc}]", leave=False)
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train(args):
    """主训练函数"""
    
    print("=" * 60)
    print("Quick Draw 涂鸦分类模型训练（优化版）")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {args.device}")
    print(f"模型类型: {args.model_type}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"每类样本: {args.samples}")
    print(f"混合精度: {'是' if args.amp else '否'}")
    print(f"数据增强: {'是' if args.augment else '否'}")
    print(f"Label Smoothing: {args.label_smoothing}")
    print("=" * 60)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU 训练")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载数据（支持数据增强）
    print("\n正在加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(
        data_dir=DATA_DIR,
        samples_per_class=args.samples,
        batch_size=args.batch_size,
        use_augmentation=args.augment
    )
    
    num_classes = len([c for c in CATEGORIES if os.path.exists(
        os.path.join(DATA_DIR, f"{c.replace(' ', '_')}_sampled.npy")
    )])
    
    if num_classes == 0:
        print("错误: 未找到任何数据文件！请先运行 download_data.py")
        return
    
    print(f"发现 {num_classes} 个类别")
    
    # 创建模型
    print("\n正在创建模型...")
    model = get_model(
        num_classes=num_classes, 
        model_type=args.model_type, 
        dropout=args.dropout,
        pretrained=True
    )
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"模型类型: {args.model_type}")
    print(f"可训练参数: {trainable:,}")
    
    # 损失函数（支持 Label Smoothing）
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # 优化器
    if args.model_type == "resnet":
        # ResNet 使用分层学习率
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'fc' in name or 'resnet.fc' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},  # backbone 用较小学习率
            {'params': head_params, 'lr': args.lr}  # head 用正常学习率
        ], weight_decay=args.weight_decay)
        print("使用分层学习率: backbone=0.1x, head=1x")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
    
    # 混合精度
    scaler = GradScaler() if args.amp else None
    
    # 创建模型保存目录
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n开始训练...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
            scaler=scaler, use_amp=args.amp
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        if args.scheduler == "cosine":
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # 打印结果
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"训练: loss={train_loss:.4f}, acc={train_acc*100:.2f}% | "
              f"验证: loss={val_loss:.4f}, acc={val_acc*100:.2f}% | "
              f"lr={current_lr:.2e} | 耗时={epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'model_type': args.model_type,
                'categories': CATEGORIES[:num_classes],
                'history': history
            }, BEST_MODEL_PATH)
            print(f"  ✓ 保存最佳模型 (验证准确率: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            
        # 早停
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n早停: 验证准确率连续 {args.patience} 轮未提升")
            break
    
    total_time = time.time() - start_time
    
    # 测试最佳模型
    print("\n" + "=" * 60)
    print("加载最佳模型进行测试...")
    
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device, desc="测试")
    
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"测试准确率: {test_acc*100:.2f}%")
    print(f"模型保存至: {BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Draw 模型训练（优化版）")
    
    # 基础参数
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 比率")
    parser.add_argument("--device", type=str, default=DEVICE, help="训练设备")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CLASS, help="每类样本数")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值，0 表示禁用")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="resnet", 
                        choices=["base", "large", "resnet"],
                        help="模型类型: base=简单CNN, large=大CNN, resnet=ResNet18预训练")
    
    # 优化参数
    parser.add_argument("--amp", action="store_true", default=True, help="使用混合精度训练")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="禁用混合精度训练")
    parser.add_argument("--augment", action="store_true", default=True, help="使用数据增强")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="禁用数据增强")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label Smoothing 系数")
    parser.add_argument("--scheduler", type=str, default="cosine", 
                        choices=["cosine", "onecycle"],
                        help="学习率调度器")
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    os.chdir(Path(__file__).parent.parent)
    
    train(args)
