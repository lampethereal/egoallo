"""
EE4D数据集到EgoAllo格式的转换工具
"""

import torch
from pathlib import Path
import sys

# 添加egoallo到路径
egoallo_src = Path(__file__).parent / "src"
if str(egoallo_src) not in sys.path:
    sys.path.insert(0, str(egoallo_src))

try:
    from egoallo.transforms import SO3, SE3
except ImportError as e:
    print(f"错误: 无法导入egoallo模块")
    print(f"请确保已安装: pip install -e .")
    print(f"原始错误: {e}")
    sys.exit(1)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """
    将rotation_6d转换为旋转矩阵
    
    Args:
        rotation_6d: [*, 6] 旋转矩阵的前两列展开
    
    Returns:
        rotation_matrix: [*, 3, 3]
    """
    batch_shape = rotation_6d.shape[:-1]
    rotation_6d = rotation_6d.reshape(-1, 6)
    
    # 提取前两列
    col1 = rotation_6d[:, :3]  # [N, 3]
    col2 = rotation_6d[:, 3:6]  # [N, 3]
    
    # Gram-Schmidt正交化
    col1 = col1 / torch.norm(col1, dim=-1, keepdim=True)
    col2 = col2 - (col1 * col2).sum(dim=-1, keepdim=True) * col1
    col2 = col2 / torch.norm(col2, dim=-1, keepdim=True)
    col3 = torch.cross(col1, col2, dim=-1)  # [N, 3]
    
    # 组装旋转矩阵 [N, 3, 3]
    rotation_matrix = torch.stack([col1, col2, col3], dim=-1)
    
    return rotation_matrix.reshape(*batch_shape, 3, 3)


def ee4d_aria_traj_to_egoallo(
    aria_traj: torch.Tensor,
    floor_height: float = 0.0,
    add_initial_frame_method: str = "extrapolate"
) -> torch.Tensor:
    """
    将EE4D的aria_traj转换为EgoAllo的Ts_world_cpf格式
    
    Args:
        aria_traj: [T, 9] rotation_6d (6) + translation (3)
        floor_height: 地面高度，会被添加到z坐标
        add_initial_frame_method: 如何添加T0帧
            - "identity": 使用单位变换
            - "first": 使用第一帧的值
            - "extrapolate": 通过外推估计（默认，最合理）
    
    Returns:
        Ts_world_cpf: [T+1, 7] wxyz (4) + xyz (3)
    """
    T = aria_traj.shape[0]
    device = aria_traj.device
    dtype = aria_traj.dtype
    
    # 1. 提取rotation_6d和translation
    rotation_6d = aria_traj[:, :6]  # [T, 6]
    translation = aria_traj[:, 6:9]  # [T, 3]
    
    # 调整地面高度
    translation = translation.clone()
    translation[:, 2] -= floor_height
    
    # 2. rotation_6d转换为旋转矩阵
    rotation_matrix = rotation_6d_to_matrix(rotation_6d)  # [T, 3, 3]
    
    # 3. 旋转矩阵转换为四元数
    so3 = SO3.from_matrix(rotation_matrix)
    wxyz = so3.wxyz  # [T, 4]
    
    # 4. 组合为SE3参数 [T, 7]
    Ts_world_cpf_1_to_T = torch.cat([wxyz, translation], dim=-1)
    
    # 5. 添加T0帧（t=-1时刻的位姿）
    if add_initial_frame_method == "identity":
        # 方法1：使用identity
        T0 = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 
            device=device, dtype=dtype
        )
    elif add_initial_frame_method == "first":
        # 方法2：重复第一帧
        T0 = Ts_world_cpf_1_to_T[0:1].clone()
    elif add_initial_frame_method == "extrapolate":
        # 方法3：通过外推估计T0（假设匀速运动）
        if T >= 2:
            # 计算从T1到T2的相对变换
            SE3_1 = SE3(Ts_world_cpf_1_to_T[0:1])
            SE3_2 = SE3(Ts_world_cpf_1_to_T[1:2])
            delta_SE3 = SE3_1.inverse() @ SE3_2
            
            # 反向应用得到T0
            SE3_0 = SE3_1 @ delta_SE3.inverse()
            T0 = SE3_0.wxyz_xyz
        else:
            # 如果只有1帧，使用identity
            T0 = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 
                device=device, dtype=dtype
            )
    else:
        raise ValueError(f"Unknown method: {add_initial_frame_method}")
    
    # 6. 拼接 [T+1, 7]
    Ts_world_cpf = torch.cat([T0, Ts_world_cpf_1_to_T], dim=0)
    
    return Ts_world_cpf


def egoallo_to_ee4d_aria_traj(
    Ts_world_cpf: torch.Tensor,
    floor_height: float = 0.0
) -> torch.Tensor:
    """
    将EgoAllo的Ts_world_cpf转换回EE4D的aria_traj格式（用于验证）
    
    Args:
        Ts_world_cpf: [T+1, 7] wxyz (4) + xyz (3)
        floor_height: 恢复地面高度
    
    Returns:
        aria_traj: [T, 9] rotation_6d (6) + translation (3)
    """
    # 去掉T0帧
    Ts_world_cpf_1_to_T = Ts_world_cpf[1:]  # [T, 7]
    
    # 提取四元数和平移
    wxyz = Ts_world_cpf_1_to_T[:, :4]
    translation = Ts_world_cpf_1_to_T[:, 4:7].clone()
    
    # 恢复地面高度
    translation[:, 2] += floor_height
    
    # 四元数转旋转矩阵
    so3 = SO3(wxyz=wxyz)
    rotation_matrix = so3.as_matrix()  # [T, 3, 3]
    
    # 旋转矩阵转rotation_6d（取前两列）
    rotation_6d = torch.cat([
        rotation_matrix[:, :, 0],  # 第1列
        rotation_matrix[:, :, 1],  # 第2列
    ], dim=-1)  # [T, 6]
    
    # 拼接
    aria_traj = torch.cat([rotation_6d, translation], dim=-1)  # [T, 9]
    
    return aria_traj


def validate_conversion(aria_traj_original: torch.Tensor, floor_height: float = 0.0):
    """验证转换的正确性"""
    print("=" * 60)
    print("验证数据转换")
    print("=" * 60)
    
    # 前向转换
    Ts_world_cpf = ee4d_aria_traj_to_egoallo(
        aria_traj_original, 
        floor_height=floor_height
    )
    print(f"✓ EE4D → EgoAllo: {aria_traj_original.shape} → {Ts_world_cpf.shape}")
    
    # 反向转换
    aria_traj_recovered = egoallo_to_ee4d_aria_traj(
        Ts_world_cpf, 
        floor_height=floor_height
    )
    print(f"✓ EgoAllo → EE4D: {Ts_world_cpf.shape} → {aria_traj_recovered.shape}")
    
    # 计算误差
    diff = torch.abs(aria_traj_original - aria_traj_recovered)
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    
    print(f"\n误差统计:")
    print(f"  最大误差: {max_error:.2e}")
    print(f"  平均误差: {mean_error:.2e}")
    
    if max_error < 1e-5:
        print(f"  ✓ 转换精度良好!")
    else:
        print(f"  ⚠ 转换存在误差，请检查")
    
    return max_error < 1e-5


# 测试代码
if __name__ == "__main__":
    print("EE4D到EgoAllo数据转换测试\n")
    
    # 创建测试数据
    T = 80
    torch.manual_seed(42)
    
    # 随机生成rotation_6d和translation
    # 这里用随机正交矩阵的前两列
    rotation_matrices = torch.randn(T, 3, 3)
    U, _, Vt = torch.linalg.svd(rotation_matrices)
    rotation_matrices = U @ Vt  # [T, 3, 3] 正交矩阵
    
    rotation_6d = torch.cat([
        rotation_matrices[:, :, 0],  # 第1列
        rotation_matrices[:, :, 1],  # 第2列
    ], dim=-1)  # [T, 6]
    
    translation = torch.randn(T, 3) * 0.1  # 小幅度平移
    
    aria_traj = torch.cat([rotation_6d, translation], dim=-1)  # [T, 9]
    
    print(f"测试数据形状: {aria_traj.shape}")
    print(f"示例数据 (第1帧):\n{aria_traj[0]}\n")
    
    # 验证转换
    success = validate_conversion(aria_traj, floor_height=-1.5)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！转换工具可以使用。")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ 测试失败，请检查转换逻辑。")
        print("=" * 60)
