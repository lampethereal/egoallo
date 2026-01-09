"""
一键对比 - UniEgoMotion vs EgoAllo

功能：
1. 在相同序列上运行两个模型
2. 自动切换conda环境（使用conda run）
3. 整理输出到统一目录
4. 生成对比报告

使用方法：
    python 一键对比_两模型.py
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime


# ============================================================================
# 配置区域
# ============================================================================

CONFIG = {
    # 项目路径
    "uniegomotion_root": r"d:\Repository\UniEgoMotion",
    "egoallo_root": r"d:\Repository\egoallo",
    
    # Conda环境名称
    "uem_env": "uem",
    "egoallo_env": "egoallo",
    
    # 对比序列（修改这里！）
    "sequences": [
        "indiana_cooking_23_5___0___513",
        # "indiana_cooking_23_5___519___2472",
        # "indiana_cooking_23_5___3030___3396",
    ],
    
    # 推理参数
    "traj_length": 64,
    "start_index": 0,
    
    # 输出目录
    "output_root": r"./model_comparison_output",
}


# ============================================================================
# 核心函数
# ============================================================================

def run_egoallo(sequence: str, output_dir: Path) -> dict:
    """运行EgoAllo推理"""
    print(f"\n{'='*80}")
    print(f"[EgoAllo] 推理: {sequence}")
    print('='*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = (
        f"conda run -n {CONFIG['egoallo_env']} "
        f"python ee4d_inference.py "
        f"--sequence {sequence} "
        f"--traj-length {CONFIG['traj_length']} "
        f"--output-dir {output_dir}"
    )
    
    print(f"命令: {cmd}")
    start = time.time()
    
    try:
        # 不捕获输出，让输出直接显示（避免缓冲区满）
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=CONFIG['egoallo_root'],
            timeout=600
        )
        
        elapsed = time.time() - start
        
        # 查找输出文件
        npz_files = list(output_dir.glob("**/*.npz"))
        
        if result.returncode == 0 and npz_files:
            print(f"✓ 成功 (用时: {elapsed:.2f}秒)")
            print(f"  输出: {npz_files[0].name}")
            return {
                'success': True,
                'time': elapsed,
                'output': str(npz_files[0]),
                'error': None
            }
        else:
            print(f"✗ 失败 (返回码: {result.returncode})")
            return {
                'success': False,
                'time': elapsed,
                'output': None,
                'error': f"Return code: {result.returncode}"
            }
    
    except subprocess.TimeoutExpired:
        print(f"✗ 超时 (>10分钟)")
        return {'success': False, 'time': 600, 'output': None, 'error': 'Timeout'}
    except Exception as e:
        print(f"✗ 异常: {e}")
        return {'success': False, 'time': 0, 'output': None, 'error': str(e)}


def run_uniegomotion(sequence: str, output_dir: Path) -> dict:
    """
    运行UniEgoMotion推理
    
    策略：创建临时Python脚本直接调用底层推理代码
    """
    print(f"\n{'='*80}")
    print(f"[UniEgoMotion] 推理: {sequence}")
    print('='*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建临时推理脚本
    temp_script = Path(CONFIG['uniegomotion_root']) / "temp_comparison_inference.py"
    output_dir_str = str(output_dir.absolute()).replace('\\', '/')
    
    script_content = f'''
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, r"{CONFIG['uniegomotion_root']}")
os.chdir(r"{CONFIG['uniegomotion_root']}")

from config.defaults import get_cfg
from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset, careful_collate_fn
from module.uem_module import UEM_Module
from utils.torch_utils import to_device
from dataset.smpl_utils import get_smpl

# 配置
sequence_name = "{sequence}"
start_idx = {CONFIG['start_index']}
output_dir = Path(r"{output_dir_str}")
output_dir.mkdir(parents=True, exist_ok=True)

# 加载配置
sys.argv = ["temp", "CONFIG", "config/uem.yaml", "TRAIN.EXP_PATH", "./exp/uem_v4b_dinov2"]
cfg = get_cfg()

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = os.path.join(cfg.TRAIN.EXP_PATH, "last.ckpt")
model = UEM_Module.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu").to(device).eval()

# 加载数据集
dataset = EE4D_Motion_Dataset(
    data_dir=cfg.DATA.DATA_DIR,
    split="val",
    repre_type=cfg.DATA.REPRE_TYPE,
    cond_traj=cfg.DATA.COND_TRAJ,
    cond_img_feat=cfg.DATA.COND_IMG_FEAT,
    cond_betas=cfg.DATA.COND_BETAS,
    window=cfg.DATA.WINDOW,
    img_feat_type=cfg.DATA.IMG_FEAT_TYPE,
)
smpl = get_smpl()

# 查找序列
found_idx = None
for idx, (seq_idx, frame_idx) in enumerate(dataset.idx_to_sidx_fidx):
    seq_name_current = dataset.seq_names[seq_idx]
    if seq_name_current == sequence_name and frame_idx == start_idx:
        found_idx = idx
        break

if found_idx is None:
    print(f"错误：找不到序列 {{sequence_name}} 起始帧 {{start_idx}}")
    print(f"可用序列: {{dataset.seq_names[:5]}}...")
    sys.exit(1)

# 推理
batch = dataset[found_idx]
batch = careful_collate_fn([batch])
batch = to_device(batch, device)

with torch.no_grad():
    y = batch["y"]
    # 采样运动
    x = model.sample(y, 1, cond_scale=None)
    # 构造预测batch
    pred_batch = {{}}
    pred_batch["y"] = batch["y"]
    pred_batch["misc"] = {{k: v for k, v in batch["misc"].items()}}
    pred_batch["pred"] = {{"motion": x.cpu()}}
    
    # 转换为完整序列
    pred_mdata = dataset.ret_to_full_sequence(pred_batch)

# 保存结果
output_file = output_dir / "infer_0.npz"

# 提取SMPL参数
pred_smpl_params = pred_mdata["smpl_params_full"][0]
pred_aria_traj_T = pred_mdata["aria_traj_T"][0]

# 保存为NPZ
np.savez(
    output_file,
    aria_traj_T=pred_aria_traj_T.cpu().numpy(),
    smpl_params=pred_smpl_params,
)

print(f"✓ 推理完成，保存至: {{output_file}}")
'''
    
    # 写入临时脚本
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 运行推理
    cmd = f"conda run -n {CONFIG['uem_env']} python {temp_script.name}"
    
    print(f"命令: {cmd}")
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=CONFIG['uniegomotion_root'],
            timeout=600
        )
        
        elapsed = time.time() - start
        
        # 清理临时文件
        temp_script.unlink(missing_ok=True)
        
        # 查找输出文件
        npz_files = list(output_dir.glob("*.npz"))
        
        if result.returncode == 0 or npz_files:
            print(f"✓ 成功 (用时: {elapsed:.2f}秒)")
            if npz_files:
                print(f"  输出: {npz_files[0].name}")
            return {
                'success': True,
                'time': elapsed,
                'output': str(npz_files[0]) if npz_files else None,
                'error': None
            }
        else:
            print(f"✗ 失败 (返回码: {result.returncode})")
            return {
                'success': False,
                'time': elapsed,
                'output': None,
                'error': f"Return code: {result.returncode}"
            }
    
    except subprocess.TimeoutExpired:
        print(f"✗ 超时 (>10分钟)")
        temp_script.unlink(missing_ok=True)
        return {'success': False, 'time': 600, 'output': None, 'error': 'Timeout'}
    except Exception as e:
        print(f"✗ 异常: {e}")
        temp_script.unlink(missing_ok=True)
        return {'success': False, 'time': 0, 'output': None, 'error': str(e)}


def save_report(results: list, output_root: Path):
    """保存对比报告"""
    
    # JSON报告
    json_path = output_root / "comparison_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': CONFIG,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    # Markdown报告
    md_lines = []
    md_lines.append(f"# UniEgoMotion vs EgoAllo 对比报告\n")
    md_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md_lines.append("---\n\n")
    
    md_lines.append("## 推理结果\n\n")
    
    for r in results:
        md_lines.append(f"### {r['sequence']}\n\n")
        
        # UniEgoMotion
        uem = r['uniegomotion']
        md_lines.append("#### UniEgoMotion\n")
        if uem['success']:
            md_lines.append(f"- ✅ 状态: 成功\n")
            md_lines.append(f"- 用时: {uem['time']:.2f}秒\n")
            if uem['output']:
                md_lines.append(f"- 输出: `{Path(uem['output']).name}`\n")
        else:
            md_lines.append(f"- ❌ 状态: 失败\n")
            md_lines.append(f"- 错误: {uem['error'][:100] if uem['error'] else 'Unknown'}\n")
        md_lines.append("\n")
        
        # EgoAllo
        ego = r['egoallo']
        md_lines.append("#### EgoAllo\n")
        if ego['success']:
            md_lines.append(f"- ✅ 状态: 成功\n")
            md_lines.append(f"- 用时: {ego['time']:.2f}秒\n")
            if ego['output']:
                md_lines.append(f"- 输出: `{Path(ego['output']).name}`\n")
        else:
            md_lines.append(f"- ❌ 状态: 失败\n")
            md_lines.append(f"- 错误: {ego['error'][:100] if ego['error'] else 'Unknown'}\n")
        md_lines.append("\n---\n\n")
    
    # 统计
    uem_success = sum(1 for r in results if r['uniegomotion']['success'])
    ego_success = sum(1 for r in results if r['egoallo']['success'])
    
    md_lines.append("## 统计汇总\n\n")
    md_lines.append(f"- UniEgoMotion: {uem_success}/{len(results)} 成功\n")
    md_lines.append(f"- EgoAllo: {ego_success}/{len(results)} 成功\n")
    
    if uem_success > 0:
        avg_uem = sum(r['uniegomotion']['time'] for r in results if r['uniegomotion']['success']) / uem_success
        md_lines.append(f"- UniEgoMotion平均用时: {avg_uem:.2f}秒\n")
    
    if ego_success > 0:
        avg_ego = sum(r['egoallo']['time'] for r in results if r['egoallo']['success']) / ego_success
        md_lines.append(f"- EgoAllo平均用时: {avg_ego:.2f}秒\n")
    
    md_path = output_root / "comparison_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)
    
    print(f"\n✓ 报告已保存:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n" + "="*80)
    print(" 一键对比 - UniEgoMotion vs EgoAllo")
    print("="*80)
    print(f"\n序列数量: {len(CONFIG['sequences'])}")
    print(f"推理帧数: {CONFIG['traj_length']}")
    print(f"输出目录: {CONFIG['output_root']}\n")
    
    # 创建输出目录
    output_root = Path(CONFIG['output_root'])
    output_root.mkdir(parents=True, exist_ok=True)
    
    uem_dir = output_root / "uniegomotion"
    ego_dir = output_root / "egoallo"
    uem_dir.mkdir(exist_ok=True)
    ego_dir.mkdir(exist_ok=True)
    
    # 批量推理
    results = []
    
    for i, seq in enumerate(CONFIG['sequences']):
        print(f"\n\n{'#'*80}")
        print(f"# 序列 {i+1}/{len(CONFIG['sequences'])}: {seq}")
        print(f"{'#'*80}")
        
        result = {'sequence': seq}
        
        # EgoAllo推理
        result['egoallo'] = run_egoallo(seq, ego_dir / seq)
        
        # UniEgoMotion推理
        result['uniegomotion'] = run_uniegomotion(seq, uem_dir / seq)
        
        results.append(result)
    
    # 保存报告
    print(f"\n\n{'='*80}")
    print(" 生成对比报告")
    print('='*80)
    save_report(results, output_root)
    
    # 汇总
    print(f"\n{'='*80}")
    print(" 对比完成")
    print('='*80)
    
    uem_ok = sum(1 for r in results if r['uniegomotion']['success'])
    ego_ok = sum(1 for r in results if r['egoallo']['success'])
    
    print(f"\n总序列: {len(results)}")
    print(f"UniEgoMotion成功: {uem_ok}/{len(results)}")
    print(f"EgoAllo成功: {ego_ok}/{len(results)}")
    
    print(f"\n详细:")
    for r in results:
        uem_icon = "✓" if r['uniegomotion']['success'] else "✗"
        ego_icon = "✓" if r['egoallo']['success'] else "✗"
        print(f"  {r['sequence']}")
        print(f"    UEM: {uem_icon}  |  EgoAllo: {ego_icon}")
    
    print(f"\n✓ 所有结果保存至: {output_root}")
    print('='*80)


if __name__ == '__main__':
    main()
