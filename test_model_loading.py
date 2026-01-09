"""测试EgoAllo模型加载"""
from egoallo.inference_utils import load_denoiser
from pathlib import Path
import torch

checkpoint_dir = Path('./egoallo_checkpoint_april13/checkpoints_3000000/')
print(f"检查点目录: {checkpoint_dir}")
print(f"存在: {checkpoint_dir.exists()}")

if checkpoint_dir.exists():
    print("加载模型...")
    model = load_denoiser(checkpoint_dir)
    print("✓ 模型加载成功!")
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"模型配置: {model.config}")
else:
    print("✗ 检查点目录不存在")
    print("请运行: bash download_checkpoint_and_data.sh")
