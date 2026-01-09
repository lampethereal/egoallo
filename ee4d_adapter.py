"""
EE4D数据适配器 - 让EgoAllo直接读取UniEgoMotion的ee_val.pt数据
绕过VRS和MPS文件加载，使用纯Python转换
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse


class MockAriaTraj:
    """模拟Aria轨迹对象，提供EgoAllo需要的接口"""
    
    def __init__(self, Ts_world_cpf: np.ndarray, timestamps_ns: np.ndarray):
        """
        Args:
            Ts_world_cpf: [T+1, 7] - SE3轨迹 (wxyz四元数 + xyz平移)
            timestamps_ns: [T] - 时间戳（纳秒）
        """
        self.Ts_world_cpf = Ts_world_cpf  # [T+1, 7]
        self.timestamps_ns = timestamps_ns  # [T]
        self.device_timestamps_ns = timestamps_ns  # 别名，某些代码可能用这个
    
    def get_timestamps_ns(self):
        """获取时间戳数组"""
        return self.timestamps_ns
    
    def __len__(self):
        return len(self.timestamps_ns)


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    rotation_6d [*, 6] 转换为旋转矩阵 [*, 3, 3]
    使用Gram-Schmidt正交化
    """
    batch_shape = rot_6d.shape[:-1]
    rot_6d = rot_6d.reshape(-1, 6)
    
    # 前两列
    col1 = rot_6d[:, :3]
    col2_raw = rot_6d[:, 3:6]
    
    # Gram-Schmidt
    col1 = col1 / torch.norm(col1, dim=1, keepdim=True)
    col2 = col2_raw - (col1 * col2_raw).sum(dim=1, keepdim=True) * col1
    col2 = col2 / torch.norm(col2, dim=1, keepdim=True)
    col3 = torch.cross(col1, col2, dim=1)
    
    rot_mat = torch.stack([col1, col2, col3], dim=2)  # [B, 3, 3]
    return rot_mat.reshape(*batch_shape, 3, 3)


def matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """
    旋转矩阵 [*, 3, 3] 转换为四元数 [*, 4] (w, x, y, z)
    使用Shepperd方法确保数值稳定性
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape: {matrix.shape}")

    batch_dim = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    batch_size = m.shape[0]

    # 对角线元素
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    trace = m00 + m11 + m22

    # 预分配四元数
    q = torch.zeros((batch_size, 4), dtype=m.dtype, device=m.device)

    # 情况1: trace > 0 (最常见，最稳定)
    mask_trace = trace > 0
    s = torch.sqrt(trace[mask_trace] + 1.0) * 2  # s = 4 * w
    q[mask_trace, 0] = 0.25 * s  # w
    q[mask_trace, 1] = (m21[mask_trace] - m12[mask_trace]) / s  # x
    q[mask_trace, 2] = (m02[mask_trace] - m20[mask_trace]) / s  # y
    q[mask_trace, 3] = (m10[mask_trace] - m01[mask_trace]) / s  # z

    # 情况2: m00最大
    mask_m00 = (~mask_trace) & (m00 > m11) & (m00 > m22)
    s = torch.sqrt(1.0 + m00[mask_m00] - m11[mask_m00] - m22[mask_m00]) * 2
    q[mask_m00, 0] = (m21[mask_m00] - m12[mask_m00]) / s
    q[mask_m00, 1] = 0.25 * s
    q[mask_m00, 2] = (m01[mask_m00] + m10[mask_m00]) / s
    q[mask_m00, 3] = (m02[mask_m00] + m20[mask_m00]) / s

    # 情况3: m11最大
    mask_m11 = (~mask_trace) & (~mask_m00) & (m11 > m22)
    s = torch.sqrt(1.0 + m11[mask_m11] - m00[mask_m11] - m22[mask_m11]) * 2
    q[mask_m11, 0] = (m02[mask_m11] - m20[mask_m11]) / s
    q[mask_m11, 1] = (m01[mask_m11] + m10[mask_m11]) / s
    q[mask_m11, 2] = 0.25 * s
    q[mask_m11, 3] = (m12[mask_m11] + m21[mask_m11]) / s

    # 情况4: m22最大
    mask_m22 = (~mask_trace) & (~mask_m00) & (~mask_m11)
    s = torch.sqrt(1.0 + m22[mask_m22] - m00[mask_m22] - m11[mask_m22]) * 2
    q[mask_m22, 0] = (m10[mask_m22] - m01[mask_m22]) / s
    q[mask_m22, 1] = (m02[mask_m22] + m20[mask_m22]) / s
    q[mask_m22, 2] = (m12[mask_m22] + m21[mask_m22]) / s
    q[mask_m22, 3] = 0.25 * s

    return q.reshape(*batch_dim, 4)


def ee4d_aria_traj_to_egoallo(
    aria_traj_6d: torch.Tensor,
    floor_height: float = 0.0,
    fps: float = 10.0
) -> Dict[str, np.ndarray]:
    """
    EE4D aria_traj [T, 9] 转换为EgoAllo所需格式
    
    Args:
        aria_traj_6d: [T, 9] - (rotation_6d [6] + translation [3])
        floor_height: 地面高度（米），用于调整Z坐标
        fps: 帧率，用于生成时间戳
    
    Returns:
        dict: {
            'Ts_world_cpf': np.ndarray [T+1, 7],  # SE3轨迹 (wxyz + xyz)
            'timestamps_ns': np.ndarray [T],       # 时间戳（纳秒）
        }
    """
    T = aria_traj_6d.shape[0]
    device = aria_traj_6d.device
    
    # 1. 分离rotation_6d和translation
    rot_6d = aria_traj_6d[:, :6]  # [T, 6]
    trans = aria_traj_6d[:, 6:9]   # [T, 3]
    
    # 2. rotation_6d → rotation_matrix → quaternion
    rot_mat = rotation_6d_to_matrix(rot_6d)  # [T, 3, 3]
    quat_wxyz = matrix_to_quaternion_wxyz(rot_mat)  # [T, 4]
    
    # 3. 调整平移（地面高度）
    trans_adjusted = trans.clone()
    trans_adjusted[:, 2] += floor_height  # Z轴偏移
    
    # 4. 组合为SE3表示 [T, 7] (wxyz + xyz)
    Ts_T = torch.cat([quat_wxyz, trans_adjusted], dim=1)  # [T, 7]
    
    # 5. 外推T0帧（向后外推）
    # 计算T0→T1的变换差分
    if T >= 2:
        # 使用T1和T2的差分来外推T0
        delta_trans = Ts_T[0, 4:7] - Ts_T[1, 4:7]  # [3]
        T0_trans = Ts_T[0, 4:7] + delta_trans
        T0_quat = Ts_T[0, :4]  # 简化：保持T1的旋转
    else:
        # 只有1帧，无法外推，简单复制
        T0_trans = Ts_T[0, 4:7] - torch.tensor([0.0, 0.0, 0.05], device=device)
        T0_quat = Ts_T[0, :4]
    
    T0 = torch.cat([T0_quat, T0_trans], dim=0)  # [7]
    Ts_world_cpf = torch.cat([T0.unsqueeze(0), Ts_T], dim=0)  # [T+1, 7]
    
    # 6. 生成时间戳（纳秒）
    interval_ns = int(1e9 / fps)  # 10 FPS → 100,000,000 ns
    timestamps_ns = np.arange(T, dtype=np.int64) * interval_ns
    
    return {
        'Ts_world_cpf': Ts_world_cpf.cpu().numpy(),  # [T+1, 7]
        'timestamps_ns': timestamps_ns,               # [T]
    }


def load_ee4d_sequence(
    ee4d_path: str,
    sequence_name: str,
    floor_height: float = 0.0
) -> MockAriaTraj:
    """
    从ee_val.pt加载指定序列，转换为MockAriaTraj对象
    
    Args:
        ee4d_path: ee_val.pt文件路径
        sequence_name: 序列名称，如 "indiana_cooking_23_5___0___513"
        floor_height: 地面高度调整（米）
    
    Returns:
        MockAriaTraj对象
    """
    print(f"[1/3] 加载EE4D数据: {ee4d_path}")
    ee_val = torch.load(ee4d_path, map_location='cpu', weights_only=False)
    
    if sequence_name not in ee_val:
        available = list(ee_val.keys())[:10]
        raise ValueError(f"序列 '{sequence_name}' 不存在！\n可用序列示例: {available}")
    
    seq_data = ee_val[sequence_name]
    aria_traj = seq_data['aria_traj']  # [T, 9]
    
    print(f"[2/3] 转换数据格式: {aria_traj.shape[0]} 帧")
    converted = ee4d_aria_traj_to_egoallo(aria_traj, floor_height=floor_height)
    
    print(f"[3/3] 创建MockAriaTraj对象")
    mock_traj = MockAriaTraj(
        Ts_world_cpf=converted['Ts_world_cpf'],
        timestamps_ns=converted['timestamps_ns']
    )
    
    print(f"✓ 数据加载完成: {len(mock_traj)} 帧, shape={mock_traj.Ts_world_cpf.shape}")
    return mock_traj


def list_available_sequences(ee4d_path: str, max_display: int = 20):
    """列出可用的EE4D序列"""
    print(f"正在加载: {ee4d_path}")
    ee_val = torch.load(ee4d_path, map_location='cpu', weights_only=False)
    
    sequences = list(ee_val.keys())
    print(f"\n总共 {len(sequences)} 个序列")
    print(f"\n前 {max_display} 个序列:")
    print("-" * 80)
    
    for i, seq_name in enumerate(sequences[:max_display]):
        seq_data = ee_val[seq_name]
        aria_traj = seq_data['aria_traj']
        num_frames = aria_traj.shape[0]
        print(f"{i+1:3d}. {seq_name:60s} ({num_frames:4d} 帧)")
    
    if len(sequences) > max_display:
        print(f"... (还有 {len(sequences) - max_display} 个序列)")


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EE4D数据适配器')
    parser.add_argument('--ee4d-path', type=str,
                        default=r'd:\Repository\UniEgoMotion\data\ee4d_motion_uniegomotion\uniegomotion\ee_val.pt',
                        help='ee_val.pt路径')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用序列')
    parser.add_argument('--sequence', type=str,
                        help='测试加载指定序列')
    parser.add_argument('--floor-height', type=float, default=0.0,
                        help='地面高度调整（米）')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_sequences(args.ee4d_path)
    elif args.sequence:
        mock_traj = load_ee4d_sequence(args.ee4d_path, args.sequence, args.floor_height)
        print(f"\n样本数据:")
        print(f"  Ts_world_cpf[0:3]:\n{mock_traj.Ts_world_cpf[:3]}")
        print(f"  timestamps_ns[0:3]: {mock_traj.timestamps_ns[:3]}")
    else:
        print("请使用 --list 列出序列，或 --sequence <名称> 加载序列")
        parser.print_help()


if __name__ == '__main__':
    main()
