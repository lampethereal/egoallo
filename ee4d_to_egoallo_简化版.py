"""
EE4D到EgoAllo数据转换 - 简化版
纯PyTorch实现，不需要egoallo包依赖
"""

import torch
import numpy as np


def normalize_vector(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """归一化向量"""
    return v / (torch.norm(v, dim=dim, keepdim=True) + 1e-8)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """
    将rotation_6d转换为旋转矩阵
    
    Args:
        rotation_6d: [*, 6] 旋转矩阵的前两列展开
    
    Returns:
        rotation_matrix: [*, 3, 3]
    """
    batch_shape = rotation_6d.shape[:-1]
    rotation_6d_flat = rotation_6d.reshape(-1, 6)
    
    # 提取前两列
    col1 = rotation_6d_flat[:, :3]  # [N, 3]
    col2 = rotation_6d_flat[:, 3:6]  # [N, 3]
    
    # Gram-Schmidt正交化
    col1 = normalize_vector(col1)
    col2 = col2 - (col1 * col2).sum(dim=-1, keepdim=True) * col1
    col2 = normalize_vector(col2)
    col3 = torch.cross(col1, col2, dim=-1)  # [N, 3]
    
    # 组装旋转矩阵 [N, 3, 3]
    rotation_matrix = torch.stack([col1, col2, col3], dim=-1)
    
    return rotation_matrix.reshape(*batch_shape, 3, 3)


def matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    旋转矩阵转四元数 (wxyz格式)
    使用稳定的算法避免数值问题
    
    Args:
        rotation_matrix: [*, 3, 3]
    
    Returns:
        quaternion: [*, 4] wxyz格式 (w, x, y, z)
    """
    batch_shape = rotation_matrix.shape[:-2]
    matrix_flat = rotation_matrix.reshape(-1, 3, 3)
    N = matrix_flat.shape[0]
    
    # 使用Shepperd's method (更数值稳定)
    trace = matrix_flat[:, 0, 0] + matrix_flat[:, 1, 1] + matrix_flat[:, 2, 2]
    
    quaternion = torch.zeros(N, 4, device=matrix_flat.device, dtype=matrix_flat.dtype)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
    quaternion[mask1, 0] = 0.25 * s1  # w
    quaternion[mask1, 1] = (matrix_flat[mask1, 2, 1] - matrix_flat[mask1, 1, 2]) / s1  # x
    quaternion[mask1, 2] = (matrix_flat[mask1, 0, 2] - matrix_flat[mask1, 2, 0]) / s1  # y
    quaternion[mask1, 3] = (matrix_flat[mask1, 1, 0] - matrix_flat[mask1, 0, 1]) / s1  # z
    
    # Case 2: m[0,0] > m[1,1] and m[0,0] > m[2,2]
    mask2 = (~mask1) & (matrix_flat[:, 0, 0] > matrix_flat[:, 1, 1]) & (matrix_flat[:, 0, 0] > matrix_flat[:, 2, 2])
    s2 = torch.sqrt(1.0 + matrix_flat[mask2, 0, 0] - matrix_flat[mask2, 1, 1] - matrix_flat[mask2, 2, 2]) * 2
    quaternion[mask2, 0] = (matrix_flat[mask2, 2, 1] - matrix_flat[mask2, 1, 2]) / s2  # w
    quaternion[mask2, 1] = 0.25 * s2  # x
    quaternion[mask2, 2] = (matrix_flat[mask2, 0, 1] + matrix_flat[mask2, 1, 0]) / s2  # y
    quaternion[mask2, 3] = (matrix_flat[mask2, 0, 2] + matrix_flat[mask2, 2, 0]) / s2  # z
    
    # Case 3: m[1,1] > m[2,2]
    mask3 = (~mask1) & (~mask2) & (matrix_flat[:, 1, 1] > matrix_flat[:, 2, 2])
    s3 = torch.sqrt(1.0 + matrix_flat[mask3, 1, 1] - matrix_flat[mask3, 0, 0] - matrix_flat[mask3, 2, 2]) * 2
    quaternion[mask3, 0] = (matrix_flat[mask3, 0, 2] - matrix_flat[mask3, 2, 0]) / s3  # w
    quaternion[mask3, 1] = (matrix_flat[mask3, 0, 1] + matrix_flat[mask3, 1, 0]) / s3  # x
    quaternion[mask3, 2] = 0.25 * s3  # y
    quaternion[mask3, 3] = (matrix_flat[mask3, 1, 2] + matrix_flat[mask3, 2, 1]) / s3  # z
    
    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + matrix_flat[mask4, 2, 2] - matrix_flat[mask4, 0, 0] - matrix_flat[mask4, 1, 1]) * 2
    quaternion[mask4, 0] = (matrix_flat[mask4, 1, 0] - matrix_flat[mask4, 0, 1]) / s4  # w
    quaternion[mask4, 1] = (matrix_flat[mask4, 0, 2] + matrix_flat[mask4, 2, 0]) / s4  # x
    quaternion[mask4, 2] = (matrix_flat[mask4, 1, 2] + matrix_flat[mask4, 2, 1]) / s4  # y
    quaternion[mask4, 3] = 0.25 * s4  # z
    
    # 归一化确保单位四元数
    quaternion = normalize_vector(quaternion, dim=-1)
    
    return quaternion.reshape(*batch_shape, 4)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    四元数转旋转矩阵
    
    Args:
        quaternion: [*, 4] wxyz格式
    
    Returns:
        rotation_matrix: [*, 3, 3]
    """
    batch_shape = quaternion.shape[:-1]
    q_flat = quaternion.reshape(-1, 4)
    
    w, x, y, z = q_flat[:, 0], q_flat[:, 1], q_flat[:, 2], q_flat[:, 3]
    
    # 计算旋转矩阵元素
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - w*z)
    r02 = 2*(x*z + w*y)
    
    r10 = 2*(x*y + w*z)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - w*x)
    
    r20 = 2*(x*z - w*y)
    r21 = 2*(y*z + w*x)
    r22 = 1 - 2*(x*x + y*y)
    
    rotation_matrix = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    
    return rotation_matrix.reshape(*batch_shape, 3, 3)


def ee4d_aria_traj_to_egoallo(
    aria_traj: torch.Tensor,
    floor_height: float = 0.0,
) -> torch.Tensor:
    """
    将EE4D的aria_traj转换为EgoAllo的Ts_world_cpf格式
    
    Args:
        aria_traj: [T, 9] rotation_6d (6) + translation (3)
        floor_height: 地面高度，会被减去（EgoAllo假设floor在z=0）
    
    Returns:
        Ts_world_cpf: [T+1, 7] wxyz (4) + xyz (3)
    """
    T = aria_traj.shape[0]
    device = aria_traj.device
    dtype = aria_traj.dtype
    
    print(f"  输入: aria_traj shape={aria_traj.shape}")
    
    # 1. 提取rotation_6d和translation
    rotation_6d = aria_traj[:, :6]  # [T, 6]
    translation = aria_traj[:, 6:9]  # [T, 3]
    
    # 调整地面高度（EgoAllo训练时假设floor在z=0）
    translation_adjusted = translation.clone()
    translation_adjusted[:, 2] -= floor_height
    
    print(f"  地面高度调整: {floor_height:.3f}m")
    print(f"  调整前z范围: [{translation[:, 2].min():.3f}, {translation[:, 2].max():.3f}]")
    print(f"  调整后z范围: [{translation_adjusted[:, 2].min():.3f}, {translation_adjusted[:, 2].max():.3f}]")
    
    # 2. rotation_6d转换为旋转矩阵
    rotation_matrix = rotation_6d_to_matrix(rotation_6d)  # [T, 3, 3]
    
    # 3. 旋转矩阵转换为四元数
    wxyz = matrix_to_quaternion(rotation_matrix)  # [T, 4]
    
    # 4. 组合为SE3参数 [T, 7]
    Ts_world_cpf_1_to_T = torch.cat([wxyz, translation_adjusted], dim=-1)
    
    # 5. 估计T0帧（通过外推，假设匀速运动）
    if T >= 2:
        # 计算第1到第2帧的变化
        delta_quat = wxyz[1] - wxyz[0]
        delta_trans = translation_adjusted[1] - translation_adjusted[0]
        
        # 外推到T0
        quat_0 = wxyz[0] - delta_quat
        trans_0 = translation_adjusted[0] - delta_trans
        
        # 重新归一化四元数
        quat_0 = normalize_vector(quat_0)
        
        T0 = torch.cat([quat_0, trans_0], dim=-1).unsqueeze(0)
    else:
        # 如果只有1帧，T0使用identity
        T0 = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 
            device=device, dtype=dtype
        )
    
    # 6. 拼接 [T+1, 7]
    Ts_world_cpf = torch.cat([T0, Ts_world_cpf_1_to_T], dim=0)
    
    print(f"  输出: Ts_world_cpf shape={Ts_world_cpf.shape}")
    
    return Ts_world_cpf


def validate_conversion():
    """验证转换的正确性"""
    print("=" * 70)
    print("验证EE4D到EgoAllo数据转换")
    print("=" * 70)
    
    # 创建测试数据
    T = 80
    torch.manual_seed(42)
    
    # 生成随机正交旋转矩阵
    rotation_matrices = torch.randn(T, 3, 3)
    U, _, Vt = torch.linalg.svd(rotation_matrices)
    rotation_matrices = U @ Vt  # [T, 3, 3] 正交矩阵
    
    # 确保是proper rotation (det=1)
    det = torch.det(rotation_matrices)
    rotation_matrices[det < 0] = -rotation_matrices[det < 0]
    
    rotation_6d = torch.cat([
        rotation_matrices[:, :, 0],  # 第1列
        rotation_matrices[:, :, 1],  # 第2列
    ], dim=-1)  # [T, 6]
    
    translation = torch.randn(T, 3) * 0.2  # 随机平移
    translation[:, 2] += 1.5  # z方向偏移
    
    aria_traj = torch.cat([rotation_6d, translation], dim=-1)  # [T, 9]
    
    print(f"\n测试数据:")
    print(f"  形状: {aria_traj.shape}")
    print(f"  平移范围: x=[{translation[:, 0].min():.2f}, {translation[:, 0].max():.2f}], "
          f"y=[{translation[:, 1].min():.2f}, {translation[:, 1].max():.2f}], "
          f"z=[{translation[:, 2].min():.2f}, {translation[:, 2].max():.2f}]")
    
    # 转换
    print(f"\n执行转换:")
    floor_height = 1.3
    Ts_world_cpf = ee4d_aria_traj_to_egoallo(aria_traj, floor_height=floor_height)
    
    # 验证四元数
    wxyz = Ts_world_cpf[:, :4]
    norms = torch.norm(wxyz, dim=-1)
    print(f"\n四元数验证:")
    print(f"  范数范围: [{norms.min():.6f}, {norms.max():.6f}] (应该接近1.0)")
    print(f"  范数偏差: max={torch.abs(norms - 1.0).max():.2e}")
    
    # 反向验证：四元数→旋转矩阵→rotation_6d
    rotation_matrix_recovered = quaternion_to_matrix(wxyz[1:])  # 去掉T0
    rotation_6d_recovered = torch.cat([
        rotation_matrix_recovered[:, :, 0],
        rotation_matrix_recovered[:, :, 1],
    ], dim=-1)
    
    diff = torch.abs(rotation_6d - rotation_6d_recovered)
    print(f"\n旋转还原精度:")
    print(f"  最大误差: {diff.max():.2e}")
    print(f"  平均误差: {diff.mean():.2e}")
    
    if diff.max() < 1e-5:
        print(f"  ✓ 旋转转换精度良好!")
    else:
        print(f"  ⚠ 旋转转换存在误差")
    
    print(f"\n" + "=" * 70)
    print(f"✓ 转换测试完成!")
    print(f"=" * 70)
    
    return Ts_world_cpf


# 测试代码
if __name__ == "__main__":
    print("\nEE4D到EgoAllo数据转换工具 - 简化版\n")
    Ts_world_cpf = validate_conversion()
    
    print(f"\n示例输出 (前3帧):")
    for i in range(min(3, Ts_world_cpf.shape[0])):
        wxyz = Ts_world_cpf[i, :4].numpy()
        xyz = Ts_world_cpf[i, 4:7].numpy()
        print(f"  帧{i}: wxyz=[{wxyz[0]:+.4f}, {wxyz[1]:+.4f}, {wxyz[2]:+.4f}, {wxyz[3]:+.4f}]  "
              f"xyz=[{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}]")
