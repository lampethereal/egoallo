"""
一键推理 - EgoAllo on EE4D数据集
类似UniEgoMotion的一条龙推理脚本
"""

import torch
import numpy as np
from pathlib import Path
from ee4d_inference import run_egoallo_inference

# ============================================================================
# 配置区域 - 修改这里的参数即可
# ============================================================================

CONFIG = {
    # 数据配置
    "ee4d_path": r"d:\Repository\UniEgoMotion\data\ee4d_motion_uniegomotion\uniegomotion\ee_val.pt",
    
    # 序列选择（可以是单个序列名，或者列表）
    "sequences": [
        "indiana_cooking_23_5___0___513",
        "indiana_cooking_23_5___519___2472",
        "indiana_cooking_23_5___3030___3396",
    ],
    
    # 推理参数
    "start_index": 0,          # 起始帧索引
    "traj_length": 64,         # 推理帧数 (EgoAllo原生支持32-128帧)
    "num_samples": 1,          # 采样数量
    "floor_height": 0.0,       # 地面高度调整（米）
    
    # 引导模式
    "guidance_mode": "no_hands",     # 选项: no_hands, aria_wrist_only, aria_hamer, hamer_wrist, hamer_reproj2
    "guidance_inner": False,         # 扩散步骤间使用引导（手部需要）
    "guidance_post": True,           # 后处理引导（减少滑步）
    
    # 模型路径
    "checkpoint_dir": "./egoallo_checkpoint_april13/checkpoints_3000000/",
    "smplh_npz_path": "./data/smplh/neutral/model.npz",
    
    # 输出配置
    "output_dir": "./ee4d_inference_output",
}


# ============================================================================
# 自动执行 - 无需修改
# ============================================================================

def main():
    """批量推理入口"""
    
    # 确保sequences是列表
    sequences = CONFIG["sequences"]
    if isinstance(sequences, str):
        sequences = [sequences]
    
    print("\n" + "=" * 80)
    print(" EgoAllo 一键推理 - EE4D数据集")
    print("=" * 80)
    print(f"\n配置信息:")
    print(f"  序列数量: {len(sequences)}")
    print(f"  推理长度: {CONFIG['traj_length']} 帧")
    print(f"  采样数量: {CONFIG['num_samples']}")
    print(f"  引导模式: {CONFIG['guidance_mode']}")
    print(f"  输出目录: {CONFIG['output_dir']}")
    print()
    
    # 批量推理
    results = []
    for i, sequence_name in enumerate(sequences):
        print(f"\n[序列 {i+1}/{len(sequences)}] {sequence_name}")
        print("-" * 80)
        
        try:
            output_file = run_egoallo_inference(
                ee4d_path=CONFIG["ee4d_path"],
                sequence_name=sequence_name,
                output_dir=CONFIG["output_dir"],
                start_index=CONFIG["start_index"],
                traj_length=CONFIG["traj_length"],
                num_samples=CONFIG["num_samples"],
                floor_height=CONFIG["floor_height"],
                guidance_mode=CONFIG["guidance_mode"],
                guidance_inner=CONFIG["guidance_inner"],
                guidance_post=CONFIG["guidance_post"],
                checkpoint_dir=CONFIG["checkpoint_dir"],
                smplh_npz_path=CONFIG["smplh_npz_path"]
            )
            results.append({
                "sequence": sequence_name,
                "status": "成功",
                "output": str(output_file)
            })
        except Exception as e:
            print(f"\n❌ 推理失败: {e}")
            results.append({
                "sequence": sequence_name,
                "status": "失败",
                "error": str(e)
            })
    
    # 输出汇总
    print("\n\n" + "=" * 80)
    print(" 推理汇总")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r["status"] == "成功")
    print(f"\n总序列数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(results) - success_count}")
    
    print("\n详细结果:")
    for i, result in enumerate(results):
        status_icon = "✓" if result["status"] == "成功" else "✗"
        print(f"  {status_icon} [{i+1}] {result['sequence']}")
        if result["status"] == "成功":
            print(f"       输出: {result['output']}")
        else:
            print(f"       错误: {result['error']}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
