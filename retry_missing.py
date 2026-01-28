"""重新下载缺失的类别"""
import os
import sys
sys.path.append('.')
from src.download_data import download_category
from src.config import DATA_DIR, SAMPLES_PER_CLASS

# 缺失的类别
missing = [
    "chair", "cooler", "crayon", "cup", "diamond",
    "dishwasher", "diving board", "dog", "fish"
]

print(f"开始重新下载 {len(missing)} 个缺失类别...")
print("=" * 50)

success = 0
for cat in missing:
    print(f"下载中: {cat}...", end=" ", flush=True)
    result = download_category(cat, DATA_DIR, SAMPLES_PER_CLASS)
    if result["status"] == "success":
        print(f"✓ {result['message']}")
        success += 1
    else:
        print(f"✗ {result['message']}")

print("=" * 50)
print(f"完成! 成功: {success}/{len(missing)}")
