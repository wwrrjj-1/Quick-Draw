"""找出缺失的类别"""
import os
import sys
sys.path.append('.')
from src.config import CATEGORIES

data_dir = 'data'
missing = []
for cat in CATEGORIES:
    filename = f"{cat.replace(' ', '_')}_sampled.npy"
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        missing.append(cat)

print(f"缺失类别 ({len(missing)}个):")
for m in missing:
    print(f"  - {m}")
