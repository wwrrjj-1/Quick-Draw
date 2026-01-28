"""
Quick Draw 数据下载脚本

从 Google Cloud Storage 下载 Quick Draw Numpy 数据集。
每个类别仅抽取前 SAMPLES_PER_CLASS 张图片以控制数据量。
"""

import os
import sys
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from src.config import CATEGORIES, DATA_DIR, SAMPLES_PER_CLASS


# Quick Draw 数据集 URL 模板
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"


def download_category(category: str, data_dir: str, samples_per_class: int) -> dict:
    """
    下载单个类别的数据
    
    Args:
        category: 类别名称
        data_dir: 数据保存目录
        samples_per_class: 每类抽取的样本数量
    
    Returns:
        包含下载状态的字典
    """
    # 处理文件名中的空格
    filename = f"{category.replace(' ', '_')}.npy"
    filepath = os.path.join(data_dir, filename)
    sampled_filepath = os.path.join(data_dir, f"{category.replace(' ', '_')}_sampled.npy")
    
    # 如果已经下载并抽样过，跳过
    if os.path.exists(sampled_filepath):
        return {"category": category, "status": "skipped", "message": "已存在"}
    
    try:
        # 构建 URL
        url = BASE_URL.format(category.replace(" ", "%20"))
        
        # 下载数据（流式下载以减少内存占用）
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # 临时保存完整文件
        temp_file = filepath + ".tmp"
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # 加载并抽样
        data = np.load(temp_file)
        
        # 抽取前 N 个样本（如果数据量不足则全部保留）
        num_samples = min(samples_per_class, len(data))
        sampled_data = data[:num_samples]
        
        # 保存抽样后的数据
        np.save(sampled_filepath, sampled_data)
        
        # 删除临时文件
        os.remove(temp_file)
        
        return {
            "category": category,
            "status": "success",
            "message": f"下载成功，抽样 {num_samples}/{len(data)} 张"
        }
        
    except requests.exceptions.RequestException as e:
        return {"category": category, "status": "error", "message": f"下载失败: {str(e)}"}
    except Exception as e:
        return {"category": category, "status": "error", "message": f"处理失败: {str(e)}"}


def download_all(categories: list = None, 
                 data_dir: str = None, 
                 samples_per_class: int = None,
                 max_workers: int = 4):
    """
    并行下载所有类别的数据
    
    Args:
        categories: 要下载的类别列表，默认为全部
        data_dir: 数据保存目录
        samples_per_class: 每类抽取的样本数量
        max_workers: 并行下载的线程数
    """
    if categories is None:
        categories = CATEGORIES
    if data_dir is None:
        data_dir = DATA_DIR
    if samples_per_class is None:
        samples_per_class = SAMPLES_PER_CLASS
    
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Quick Draw 数据下载器")
    print(f"=" * 60)
    print(f"类别数量: {len(categories)}")
    print(f"每类样本: {samples_per_class}")
    print(f"预估大小: {len(categories) * samples_per_class * 784 / 1024 / 1024:.1f} MB")
    print(f"保存目录: {os.path.abspath(data_dir)}")
    print(f"并行线程: {max_workers}")
    print(f"=" * 60)
    
    # 统计结果
    success_count = 0
    skip_count = 0
    error_count = 0
    errors = []
    
    # 使用线程池并行下载
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有下载任务
        futures = {
            executor.submit(download_category, cat, data_dir, samples_per_class): cat 
            for cat in categories
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(categories), desc="下载进度", unit="类") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result["status"] == "success":
                    success_count += 1
                    pbar.set_postfix_str(f"✓ {result['category']}")
                elif result["status"] == "skipped":
                    skip_count += 1
                else:
                    error_count += 1
                    errors.append(result)
                    pbar.set_postfix_str(f"✗ {result['category']}")
                
                pbar.update(1)
    
    # 打印汇总
    print(f"\n" + "=" * 60)
    print(f"下载完成！")
    print(f"=" * 60)
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {error_count}")
    
    if errors:
        print(f"\n失败列表:")
        for err in errors:
            print(f"  - {err['category']}: {err['message']}")
    
    return success_count, skip_count, error_count


def test_download():
    """测试下载单个类别"""
    print("测试模式：下载 'cat' 类别...")
    result = download_category("cat", DATA_DIR, SAMPLES_PER_CLASS)
    print(f"结果: {result}")
    
    # 验证文件
    filepath = os.path.join(DATA_DIR, "cat_sampled.npy")
    if os.path.exists(filepath):
        data = np.load(filepath)
        print(f"验证成功: 形状={data.shape}, 类型={data.dtype}")
        return True
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Draw 数据下载器")
    parser.add_argument("--test", action="store_true", help="测试模式，只下载一个类别")
    parser.add_argument("--workers", type=int, default=4, help="并行下载线程数")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CLASS, help="每类样本数")
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    os.chdir(Path(__file__).parent.parent)
    
    if args.test:
        test_download()
    else:
        download_all(max_workers=args.workers, samples_per_class=args.samples)
