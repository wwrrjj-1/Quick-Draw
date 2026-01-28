"""
Quick Draw 模型评估脚本

评估训练好的模型，生成可视化报告。
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from src.config import DATA_DIR, MODEL_DIR, BEST_MODEL_PATH, DEVICE, BATCH_SIZE, SAMPLES_PER_CLASS
from src.dataset import get_dataloaders
from src.model import get_model


def load_model(model_path: str, device: torch.device):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    num_classes = checkpoint.get('num_classes', 345)
    categories = checkpoint.get('categories', [])
    model_type = checkpoint.get('model_type', 'base')  # 获取模型类型
    
    model = get_model(num_classes=num_classes, model_type=model_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功!")
    print(f"  - 模型类型: {model_type}")
    print(f"  - 类别数: {num_classes}")
    print(f"  - 最佳验证准确率: {checkpoint.get('val_acc', 0)*100:.2f}%")
    
    return model, categories


def evaluate(model, test_loader, device, categories):
    """
    完整评估模型
    
    Returns:
        all_preds, all_labels, accuracy
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="评估中"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    
    return all_preds, all_labels, accuracy


def plot_confusion_matrix(y_true, y_pred, categories, save_path=None, top_n=20):
    """
    绘制混淆矩阵（只显示 top_n 个最常见的类别）
    """
    # 如果类别太多，只显示 top_n
    if len(categories) > top_n:
        # 找出预测最多的类别
        unique, counts = np.unique(y_true, return_counts=True)
        top_indices = unique[np.argsort(counts)[-top_n:]]
        
        # 过滤数据
        mask = np.isin(y_true, top_indices)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # 重新映射标签
        label_map = {old: new for new, old in enumerate(top_indices)}
        y_true_mapped = np.array([label_map[l] for l in y_true_filtered])
        y_pred_mapped = np.array([label_map.get(l, -1) for l in y_pred_filtered])
        
        # 过滤无效预测
        valid_mask = y_pred_mapped >= 0
        y_true_mapped = y_true_mapped[valid_mask]
        y_pred_mapped = y_pred_mapped[valid_mask]
        
        display_categories = [categories[i] if i < len(categories) else f"class_{i}" 
                             for i in top_indices]
    else:
        y_true_mapped = y_true
        y_pred_mapped = y_pred
        display_categories = categories
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_mapped, y_pred_mapped)
    
    # 绘制
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=display_categories,
                yticklabels=display_categories)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title(f'混淆矩阵 (Top {len(display_categories)} 类别)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"混淆矩阵已保存: {save_path}")
    
    plt.show()


def plot_sample_predictions(model, dataset, device, categories, 
                            num_samples=16, save_path=None):
    """
    可视化样本预测结果
    """
    model.eval()
    
    # 随机抽样
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            img_batch = img.unsqueeze(0).to(device)
            
            output = model(img_batch)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_idx].item()
            
            # 获取类别名称
            true_name = categories[label] if label < len(categories) else f"class_{label}"
            pred_name = categories[pred_idx] if pred_idx < len(categories) else f"class_{pred_idx}"
            
            # 绘制图像
            img_np = img.squeeze().numpy()
            axes[i].imshow(img_np, cmap='gray')
            
            # 设置标题颜色
            color = 'green' if pred_idx == label else 'red'
            axes[i].set_title(f"真: {true_name}\n预: {pred_name} ({pred_prob*100:.1f}%)",
                            fontsize=8, color=color)
            axes[i].axis('off')
    
    plt.suptitle('样本预测结果 (绿色=正确, 红色=错误)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"预测可视化已保存: {save_path}")
    
    plt.show()


def plot_training_history(model_path: str, save_path=None):
    """绘制训练历史曲线"""
    checkpoint = torch.load(model_path, map_location='cpu')
    history = checkpoint.get('history', None)
    
    if history is None:
        print("模型检查点中没有训练历史")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练/验证损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    train_acc = [a * 100 for a in history['train_acc']]
    val_acc = [a * 100 for a in history['val_acc']]
    
    ax2.plot(epochs, train_acc, 'b-', label='训练准确率')
    ax2.plot(epochs, val_acc, 'r-', label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('训练/验证准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"训练曲线已保存: {save_path}")
    
    plt.show()


def main(args):
    """主评估函数"""
    print("=" * 60)
    print("Quick Draw 模型评估")
    print("=" * 60)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载模型
    model, categories = load_model(args.model_path, device)
    
    # 如果没有保存类别信息，使用默认
    if not categories:
        from src.config import CATEGORIES
        categories = CATEGORIES
    
    # 加载数据
    print("\n加载测试数据...")
    _, _, test_loader, dataset = get_dataloaders(
        data_dir=DATA_DIR,
        samples_per_class=args.samples,
        batch_size=args.batch_size
    )
    
    # 评估
    print("\n开始评估...")
    preds, labels, accuracy = evaluate(model, test_loader, device, categories)
    
    print(f"\n测试准确率: {accuracy * 100:.2f}%")
    
    # 创建输出目录
    output_dir = os.path.join(MODEL_DIR, "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_history(args.model_path, 
                         save_path=os.path.join(output_dir, "training_history.png"))
    
    # 绘制混淆矩阵
    print("\n绘制混淆矩阵...")
    plot_confusion_matrix(labels, preds, categories, 
                         save_path=os.path.join(output_dir, "confusion_matrix.png"))
    
    # 绘制样本预测
    print("\n绘制样本预测...")
    plot_sample_predictions(model, dataset, device, categories,
                           save_path=os.path.join(output_dir, "sample_predictions.png"))
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print(f"结果保存至: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Draw 模型评估")
    
    parser.add_argument("--model_path", type=str, default=BEST_MODEL_PATH,
                        help="模型检查点路径")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="批次大小")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CLASS, help="每类样本数")
    parser.add_argument("--device", type=str, default=DEVICE, help="设备")
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    os.chdir(Path(__file__).parent.parent)
    
    main(args)
