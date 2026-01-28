"""
Quick Draw PyTorch Dataset

加载已下载的 Quick Draw 数据，提供 PyTorch 训练接口。
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, List, Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from src.config import CATEGORIES, DATA_DIR, SAMPLES_PER_CLASS, IMG_SIZE
from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO


class QuickDrawDataset(Dataset):
    """
    Quick Draw 数据集
    
    加载多个类别的 .npy 文件，合并为统一的数据集。
    """
    
    def __init__(self, 
                 data_dir: str = None,
                 categories: List[str] = None,
                 samples_per_class: int = None,
                 transform=None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            categories: 要加载的类别列表
            samples_per_class: 每类加载的样本数量
            transform: 数据增强变换
        """
        if data_dir is None:
            data_dir = DATA_DIR
        if categories is None:
            categories = CATEGORIES
        if samples_per_class is None:
            samples_per_class = SAMPLES_PER_CLASS
            
        self.data_dir = data_dir
        self.categories = categories
        self.samples_per_class = samples_per_class
        self.transform = transform
        
        # 类别到索引的映射
        self.class_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        self.idx_to_class = {idx: cat for cat, idx in self.class_to_idx.items()}
        
        # 加载数据
        self.data, self.labels = self._load_data()
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载并合并所有类别的数据"""
        all_data = []
        all_labels = []
        loaded_count = 0
        missing_categories = []
        
        print(f"正在加载 Quick Draw 数据集...")
        
        for idx, category in enumerate(self.categories):
            # 尝试找到对应的文件
            filename = f"{category.replace(' ', '_')}_sampled.npy"
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                missing_categories.append(category)
                continue
            
            # 加载数据
            data = np.load(filepath)
            
            # 限制样本数量
            num_samples = min(self.samples_per_class, len(data))
            data = data[:num_samples]
            
            # 创建标签
            labels = np.full(num_samples, idx, dtype=np.int64)
            
            all_data.append(data)
            all_labels.append(labels)
            loaded_count += 1
        
        if missing_categories:
            print(f"警告: {len(missing_categories)} 个类别文件缺失")
            if len(missing_categories) <= 10:
                print(f"  缺失类别: {missing_categories}")
        
        if not all_data:
            raise ValueError(f"未找到任何数据文件！请先运行 download_data.py")
        
        # 合并数据
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"加载完成: {loaded_count}/{len(self.categories)} 个类别，"
              f"共 {len(combined_data)} 个样本")
        
        return combined_data, combined_labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # 获取图像数据
        img = self.data[idx].astype(np.float32)
        
        # 归一化到 [0, 1]
        img = img / 255.0
        
        # 重塑为 (1, 28, 28) 的张量
        img = img.reshape(1, IMG_SIZE, IMG_SIZE)
        
        # 转换为 PyTorch 张量
        img = torch.from_numpy(img)
        
        # 应用数据增强
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        
        return img, label
    
    def get_class_name(self, idx: int) -> str:
        """根据类别索引获取类别名称"""
        return self.idx_to_class.get(idx, "unknown")
    
    def get_class_idx(self, name: str) -> int:
        """根据类别名称获取类别索引"""
        return self.class_to_idx.get(name, -1)


def get_dataloaders(data_dir: str = None,
                    categories: List[str] = None,
                    samples_per_class: int = None,
                    batch_size: int = 256,
                    num_workers: int = 4,
                    train_ratio: float = None,
                    val_ratio: float = None,
                    seed: int = 42,
                    use_augmentation: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader, QuickDrawDataset]:
    """
    获取训练、验证、测试数据加载器
    
    Args:
        data_dir: 数据目录
        categories: 类别列表
        samples_per_class: 每类样本数
        batch_size: 批次大小
        num_workers: 数据加载线程数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
        use_augmentation: 是否使用数据增强
    
    Returns:
        (train_loader, val_loader, test_loader, dataset)
    """
    import torchvision.transforms as T
    
    if train_ratio is None:
        train_ratio = TRAIN_RATIO
    if val_ratio is None:
        val_ratio = VAL_RATIO
    
    # 数据增强 transform（仅用于训练集）
    if use_augmentation:
        train_transform = T.Compose([
            T.RandomRotation(15),  # 随机旋转 ±15°
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 随机平移 10%
                scale=(0.9, 1.1),  # 随机缩放 90%-110%
            ),
        ])
        print("数据增强: 已启用 (旋转、平移、缩放)")
    else:
        train_transform = None
    
    # 创建完整数据集（无增强，用于划分）
    dataset = QuickDrawDataset(
        data_dir=data_dir,
        categories=categories,
        samples_per_class=samples_per_class,
        transform=None  # 先不应用 transform
    )
    
    # 计算划分数量
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    print(f"数据划分: 训练={train_size}, 验证={val_size}, 测试={test_size}")
    
    # 随机划分数据集
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # 创建带数据增强的训练数据集包装器
    class AugmentedDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __len__(self):
            return len(self.subset)
        
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
    
    if use_augmentation:
        train_dataset = AugmentedDataset(train_dataset, train_transform)
    
    # 创建数据加载器
    # Windows 下 num_workers > 0 可能有问题，建议设为 0
    if sys.platform == "win32":
        num_workers = 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset


if __name__ == "__main__":
    """测试数据集加载"""
    # 切换到项目根目录
    os.chdir(Path(__file__).parent.parent)
    
    # 测试加载少量类别
    test_categories = ["cat", "dog", "apple"]
    
    try:
        dataset = QuickDrawDataset(
            data_dir=DATA_DIR,
            categories=test_categories,
            samples_per_class=100
        )
        
        print(f"\n数据集大小: {len(dataset)}")
        
        # 测试 __getitem__
        img, label = dataset[0]
        print(f"样本形状: {img.shape}")
        print(f"标签: {label} -> {dataset.get_class_name(label)}")
        
        # 测试数据加载器
        train_loader, val_loader, test_loader, _ = get_dataloaders(
            data_dir=DATA_DIR,
            categories=test_categories,
            samples_per_class=100,
            batch_size=32
        )
        
        print(f"\n训练批次数: {len(train_loader)}")
        
        # 获取一个批次
        images, labels = next(iter(train_loader))
        print(f"批次形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        
        print("\n✓ 数据集测试通过！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        print("请确保已运行 download_data.py 下载数据")
