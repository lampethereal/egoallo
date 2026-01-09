# EgoAllo 数据格式和接口说明

## 核心数据格式

### 输入格式：Ts_world_cpf

**格式**: `Tensor[T+1, 7]`（注意T+1！）

**含义**: SE(3)变换的紧凑表示
- 前4维: wxyz 四元数（旋转）
- 后3维: xyz 平移（米）

**坐标系**:
- `world`: 世界坐标系
- `cpf`: CPF (Central Pupil Frame) - Aria眼镜的中心瞳孔坐标系

**示例数据**:
```python
Ts_world_cpf[0] = [
    0.9848, -0.0145, 0.1730, 0.0044,  # wxyz 四元数
    0.1139, -0.4472, 0.1249            # xyz 平移 (米)
]
```

### 网络实际输入：T_cpf_tm1_cpf_t

**格式**: `Tensor[T, 6]` (相对变换的切空间表示)

**计算方式**:
```python
# 从 Ts_world_cpf [T+1, 7] 计算相对变换
T_cpf_tm1_cpf_t = (
    SE3(Ts_world_cpf[:-1]).inverse() @ SE3(Ts_world_cpf[1:])
).wxyz_xyz  # [T, 7]

# 然后网络内部会转换为切空间表示 [T, 6]
```

**物理含义**: 从t-1时刻的CPF坐标系到t时刻的CPF坐标系的相对变换

---

## 输出格式：EgoDenoiseTraj

输出是一个包含完整身体姿态的结构：

```python
class EgoDenoiseTraj:
    body_quat_6d: Tensor[..., T, 21, 6]     # 身体21个关节的6D旋转
    left_hand_quat_6d: Tensor[..., T, 15, 6]  # 左手15个关节
    right_hand_quat_6d: Tensor[..., T, 15, 6] # 右手15个关节
    betas: Tensor[..., 10]                  # SMPL-H形状参数
    contacts: Tensor[..., T, 8]             # 接触标签(脚、手、膝盖)
```

### 保存的NPZ文件格式

```python
{
    "Ts_world_cpf": [T, 7],           # 头部轨迹(不是T+1!)
    "Ts_world_root": [T, 7],          # 身体根节点轨迹
    "body_quats": [T, 21, 4],         # wxyz四元数
    "left_hand_quats": [T, 15, 4],
    "right_hand_quats": [T, 15, 4],
    "contacts": [T, 8],
    "betas": [10],
    "frame_nums": [T],
    "timestamps_ns": [T],
}
```

---

## EE4D到EgoAllo的数据转换

### EE4D格式：aria_traj

```python
# EE4D格式 [T, 9]
aria_traj = [
    r11, r12, r13,  # rotation_6d 前3维（第1列）
    r21, r22, r23,  # rotation_6d 后3维（第2列）  
    tx, ty, tz      # 平移
]
```

### 转换函数

```python
def ee4d_to_egoallo_transform(aria_traj: torch.Tensor) -> torch.Tensor:
    """
    将EE4D的aria_traj [T, 9] 转换为EgoAllo的Ts_world_cpf [T+1, 7]
    
    Args:
        aria_traj: [T, 9] rotation_6d (6维) + translation (3维)
    
    Returns:
        Ts_world_cpf: [T+1, 7] wxyz (4维) + xyz (3维)
    """
    from egoallo.transforms import SO3, SE3
    
    T = aria_traj.shape[0]
    
    # 1. 提取rotation_6d和translation
    rotation_6d = aria_traj[:, :6]  # [T, 6]
    translation = aria_traj[:, 6:9]  # [T, 3]
    
    # 2. rotation_6d转换为旋转矩阵
    # rotation_6d是旋转矩阵的前两列
    col1 = rotation_6d[:, :3]  # [T, 3]
    col2 = rotation_6d[:, 3:6]  # [T, 3]
    
    # Gram-Schmidt正交化
    col1 = col1 / torch.norm(col1, dim=-1, keepdim=True)
    col2 = col2 - (col1 * col2).sum(dim=-1, keepdim=True) * col1
    col2 = col2 / torch.norm(col2, dim=-1, keepdim=True)
    col3 = torch.cross(col1, col2, dim=-1)  # [T, 3]
    
    # 组装旋转矩阵 [T, 3, 3]
    rotation_matrix = torch.stack([col1, col2, col3], dim=-1)
    
    # 3. 旋转矩阵转换为四元数
    so3 = SO3.from_matrix(rotation_matrix)
    wxyz = so3.wxyz  # [T, 4]
    
    # 4. 组合为SE3参数 [T, 7]
    Ts_world_cpf_T = torch.cat([wxyz, translation], dim=-1)
    
    # 5. 添加初始帧（T=0时刻的位姿）
    # 假设第0帧是identity或第1帧的前一帧
    T0 = Ts_world_cpf_T[0:1].clone()
    
    # 方法1：使用identity作为初始帧
    # T0 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 
    #                   device=aria_traj.device, dtype=aria_traj.dtype)
    
    # 方法2：通过外推估计T0（假设匀速运动）
    if T >= 2:
        # 使用前两帧估计速度
        delta_SE3 = SE3(Ts_world_cpf_T[0:1]).inverse() @ SE3(Ts_world_cpf_T[1:2])
        T0_SE3 = SE3(Ts_world_cpf_T[0:1]) @ delta_SE3.inverse()
        T0 = T0_SE3.wxyz_xyz
    
    Ts_world_cpf = torch.cat([T0, Ts_world_cpf_T], dim=0)  # [T+1, 7]
    
    return Ts_world_cpf
```

---

## 关键差异总结

| 特性 | EE4D (UniEgoMotion) | EgoAllo |
|------|---------------------|---------|
| **输入长度** | T帧 | T+1帧（需要额外的t-1帧） |
| **旋转表示** | rotation_6d [6] | 四元数 wxyz [4] |
| **数据格式** | [T, 9] | [T+1, 7] |
| **坐标系** | 世界坐标系 | 世界坐标系 |
| **手部模型** | PCA压缩 [12] | 完整关节 [15] |
| **SMPL版本** | SMPL-X | SMPL-H |
| **采样频率** | 10 FPS | 30 FPS（可配置） |
| **窗口大小** | 80帧固定 | 128帧（可变） |

---

## 推理流程对比

### UniEgoMotion
```python
1. 加载 ee_val.pt → aria_traj [T, 9]
2. 归一化
3. 扩散模型推理 → motion [T, 243]
4. 反归一化
5. 转换为SMPL-X参数
```

### EgoAllo
```python
1. 加载 aria_traj [T, 9]
2. 转换为 Ts_world_cpf [T+1, 7]
3. 计算相对变换 T_cpf_tm1_cpf_t [T, 6]
4. 扩散模型推理 → EgoDenoiseTraj
5. (可选) Guidance优化（需要HaMeR手部检测）
6. 保存SMPL-H参数
```

---

## 重要注意事项

### 1. 时间对齐
- EgoAllo需要T+1帧是因为它计算**相对变换**
- 第0帧是"前一帧"，用于计算第1帧的相对运动
- 实际输出还是T帧的运动

### 2. 坐标系
- 两个模型都使用世界坐标系
- EgoAllo会自动调整地面高度（floor_z）
- 需要确保EE4D的坐标系与世界坐标系一致

### 3. SMPL模型差异
- UniEgoMotion: SMPL-X (21身体 + 左右手各15 + 脸部)
- EgoAllo: SMPL-H (21身体 + 左右手各15)
- 身体部分兼容，但手部参数化不同

### 4. 评估指标
- 两个模型可以用相同的3D关节点误差（MPJPE）
- 需要统一到相同的关节定义
- 手部评估可能需要特殊处理
