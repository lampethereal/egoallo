"""
EgoAllo EE4D推理脚本 - 直接从UniEgoMotion的ee_val.pt数据进行推理
绕过VRS/MPS加载，使用纯Python数据转换
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import time

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.inference_utils import load_denoiser
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3
from ee4d_adapter import load_ee4d_sequence


def run_egoallo_inference(
    ee4d_path: str,
    sequence_name: str,
    output_dir: str,
    start_index: int = 0,
    traj_length: int = 64,
    num_samples: int = 1,
    floor_height: float = 0.0,
    guidance_mode: str = "no_hands",
    guidance_inner: bool = False,
    guidance_post: bool = True,
    checkpoint_dir: str = "./egoallo_checkpoint_april13/checkpoints_3000000/",
    smplh_npz_path: str = "./data/smplh/neutral/model.npz"
):
    """
    在EE4D序列上运行EgoAllo推理
    
    Args:
        ee4d_path: ee_val.pt文件路径
        sequence_name: 序列名称
        output_dir: 输出目录
        start_index: 起始帧索引
        traj_length: 推理帧数
        num_samples: 采样数量
        floor_height: 地面高度调整
        guidance_mode: 引导模式 (no_hands, aria_wrist_only等)
        guidance_inner: 扩散步骤间是否使用引导优化
        guidance_post: 扩散采样后是否使用引导优化
        checkpoint_dir: EgoAllo模型路径
        smplh_npz_path: SMPL-H模型路径
    """
    device = torch.device("cuda")
    
    print("=" * 80)
    print("EgoAllo EE4D推理")
    print("=" * 80)
    
    # 1. 加载EE4D数据
    print(f"\n[步骤1/5] 加载EE4D序列: {sequence_name}")
    mock_traj = load_ee4d_sequence(ee4d_path, sequence_name, floor_height)
    total_frames = len(mock_traj)
    print(f"  总帧数: {total_frames}")
    print(f"  起始索引: {start_index}")
    print(f"  推理长度: {traj_length}")
    
    # 检查帧范围
    if start_index + traj_length > total_frames:
        print(f"  警告: 请求范围 [{start_index}, {start_index+traj_length}] 超出总帧数 {total_frames}")
        traj_length = total_frames - start_index
        print(f"  调整推理长度为: {traj_length}")
    
    # 2. 准备Ts_world_cpf数据
    print(f"\n[步骤2/5] 准备轨迹数据")
    # 注意：需要T+1个pose用于相对变换计算
    Ts_world_cpf_np = mock_traj.Ts_world_cpf[start_index : start_index + traj_length + 1]
    Ts_world_cpf = torch.from_numpy(Ts_world_cpf_np).to(device)  # [T+1, 7]
    
    timestamps_ns = mock_traj.timestamps_ns[start_index : start_index + traj_length]
    pose_timestamps_sec = timestamps_ns / 1e9
    
    print(f"  Ts_world_cpf shape: {Ts_world_cpf.shape}")
    print(f"  时间戳范围: {timestamps_ns[0]} - {timestamps_ns[-1]} ns")
    
    # 3. 加载模型
    print(f"\n[步骤3/5] 加载EgoAllo模型")
    denoiser_network = load_denoiser(Path(checkpoint_dir)).to(device)
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)
    print(f"  ✓ 模型加载完成")
    
    # 4. 创建输出目录
    output_path = Path(output_dir) / sequence_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n[步骤4/5] 输出目录: {output_path}")
    
    # 5. 执行推理
    print(f"\n[步骤5/5] 开始推理...")
    print(f"  采样数: {num_samples}")
    print(f"  引导模式: {guidance_mode}")
    print(f"  内部引导: {guidance_inner}")
    print(f"  后处理引导: {guidance_post}")
    
    start_time = time.time()
    
    # 调用EgoAllo推理（使用和3_aria_inference.py相同的接口）
    traj = run_sampling_with_stitching(
        denoiser_network,
        body_model=body_model,
        guidance_mode=guidance_mode,
        guidance_inner=guidance_inner,
        guidance_post=guidance_post,
        Ts_world_cpf=Ts_world_cpf,
        hamer_detections=None,  # EE4D数据没有HaMeR检测
        aria_detections=None,   # EE4D数据没有Aria手部检测
        num_samples=num_samples,
        device=device,
        floor_z=floor_height,
    )
    
    inference_time = time.time() - start_time
    print(f"\n  ✓ 推理完成! 用时: {inference_time:.2f}秒")
    
    # 6. 保存结果
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_name = f"{timestamp}_{start_index}-{start_index + traj_length}"
    output_file = output_path / (save_name + ".npz")
    
    # 应用body模型获取姿态
    posed = traj.apply_to_body(body_model)
    Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        posed, Ts_world_cpf[..., 1:, :]
    )
    
    print(f"\n  保存至: {output_file}")
    np.savez(
        output_file,
        Ts_world_cpf=Ts_world_cpf[1:, :].cpu().numpy(),
        Ts_world_root=Ts_world_root.cpu().numpy(),
        body_quats=posed.local_quats[..., :21, :].cpu().numpy(),
        left_hand_quats=posed.local_quats[..., 21:36, :].cpu().numpy(),
        right_hand_quats=posed.local_quats[..., 36:51, :].cpu().numpy(),
        contacts=traj.contacts.cpu().numpy(),
        betas=traj.betas.cpu().numpy(),
        frame_nums=np.arange(start_index, start_index + traj_length),
        timestamps_ns=timestamps_ns,
    )
    
    print(f"  数据形状:")
    print(f"    Ts_world_cpf: {Ts_world_cpf[1:, :].shape}")
    print(f"    body_quats: {posed.local_quats[..., :21, :].shape}")
    print(f"    left_hand_quats: {posed.local_quats[..., 21:36, :].shape}")
    print(f"    right_hand_quats: {posed.local_quats[..., 36:51, :].shape}")
    print(f"    contacts: {traj.contacts.shape}")
    print(f"    betas: {traj.betas.shape}")
    
    print("\n" + "=" * 80)
    print(f"✓ 推理完成！输出保存在: {output_file}")
    print("=" * 80)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='EgoAllo EE4D推理 - 在UniEgoMotion数据集上运行EgoAllo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据参数
    parser.add_argument('--ee4d-path', type=str,
                        default=r'd:\Repository\UniEgoMotion\data\ee4d_motion_uniegomotion\uniegomotion\ee_val.pt',
                        help='ee_val.pt文件路径')
    parser.add_argument('--sequence', type=str, required=True,
                        help='序列名称，如: indiana_cooking_23_5___0___513')
    parser.add_argument('--floor-height', type=float, default=0.0,
                        help='地面高度调整（米）')
    
    # 推理参数
    parser.add_argument('--start-index', type=int, default=0,
                        help='起始帧索引')
    parser.add_argument('--traj-length', type=int, default=64,
                        help='推理帧数')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='采样数量')
    parser.add_argument('--guidance-mode', type=str, default='no_hands',
                        choices=['no_hands', 'aria_wrist_only', 'aria_hamer', 'hamer_wrist', 'hamer_reproj2'],
                        help='手部引导模式')
    parser.add_argument('--guidance-inner', action='store_true',
                        help='在扩散步骤间使用引导优化')
    parser.add_argument('--guidance-post', action='store_true', default=True,
                        help='在扩散采样后使用引导优化')
    
    # 模型参数
    parser.add_argument('--checkpoint-dir', type=str,
                        default='./egoallo_checkpoint_april13/checkpoints_3000000/',
                        help='EgoAllo模型路径')
    parser.add_argument('--smplh-npz-path', type=str,
                        default='./data/smplh/neutral/model.npz',
                        help='SMPL-H模型路径')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str,
                        default='./ee4d_inference_output',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 执行推理
    run_egoallo_inference(
        ee4d_path=args.ee4d_path,
        sequence_name=args.sequence,
        output_dir=args.output_dir,
        start_index=args.start_index,
        traj_length=args.traj_length,
        num_samples=args.num_samples,
        floor_height=args.floor_height,
        guidance_mode=args.guidance_mode,
        guidance_inner=args.guidance_inner,
        guidance_post=args.guidance_post,
        checkpoint_dir=args.checkpoint_dir,
        smplh_npz_path=args.smplh_npz_path
    )


if __name__ == '__main__':
    main()
