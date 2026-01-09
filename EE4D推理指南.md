# EgoAllo EE4D推理指南

## 简介

本指南介绍如何使用EgoAllo在UniEgoMotion的EE4D数据集上进行推理，实现两个模型的对比。

## 环境准备

### 1. Conda环境

```bash
conda activate egoallo
```

**重要**: 必须使用CUDA 12.8 + PyTorch 2.11 nightly

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2. 验证环境

```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
# 应输出: CUDA可用: True
```

---

## 快速开始

### 方式1: 一键推理脚本

**最简单的方式** - 编辑配置后直接运行

```bash
python 一键推理_EE4D.py
```

**配置文件** (修改`一键推理_EE4D.py`开头的CONFIG部分):

```python
CONFIG = {
    # 序列选择
    "sequences": [
        "indiana_cooking_23_5___0___513",
        "indiana_cooking_23_5___519___2472",
        "indiana_cooking_23_5___3030___3396",
    ],
    
    # 推理参数
    "traj_length": 64,         # 推理帧数 (32-128)
    "num_samples": 1,          # 采样数量
    
    # 引导模式
    "guidance_mode": "no_hands",  # 不使用手部引导（最快）
    "guidance_inner": False,      # 不在扩散步骤间引导
    "guidance_post": True,        # 后处理引导（减少滑步）
    
    # 输出
    "output_dir": "./ee4d_inference_output",
}
```

### 方式2: 命令行推理

**单个序列推理**

```bash
python ee4d_inference.py \
  --sequence indiana_cooking_23_5___0___513 \
  --traj-length 64 \
  --num-samples 1 \
  --guidance-mode no_hands
```

**查看所有可用序列**

```bash
python ee4d_adapter.py --list
```

**测试单个序列数据加载**

```bash
python ee4d_adapter.py --sequence indiana_cooking_23_5___0___513
```

---

## 参数说明

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sequence` | 必填 | EE4D序列名称 |
| `--start-index` | 0 | 起始帧索引 |
| `--traj-length` | 64 | 推理帧数 (建议32-128) |
| `--num-samples` | 1 | 采样数量 |
| `--floor-height` | 0.0 | 地面高度调整（米） |

### 引导模式

| 模式 | 说明 | 速度 | 手部精度 |
|------|------|------|----------|
| `no_hands` | 不使用手部引导 | 最快 (7秒/64帧) | 低 |
| `aria_wrist_only` | 仅Aria手腕引导 | 中等 | 中等 |
| `aria_hamer` | Aria + HaMeR引导 | 慢 | 高 |
| `hamer_wrist` | HaMeR手腕引导 | 慢 | 高 |
| `hamer_reproj2` | HaMeR重投影引导 | 最慢 | 最高 |

**注意**: EE4D数据没有HaMeR检测，建议使用`no_hands`

### 引导优化

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--guidance-inner` | False | 扩散步骤间使用引导（手部需要） |
| `--guidance-post` | True | 后处理引导（减少滑步） |

**推荐配置**:
- 快速测试: `--guidance-post` (7秒/64帧)
- 平衡质量: `--guidance-inner --guidance-post` (15秒/64帧)

### 模型路径

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint-dir` | `./egoallo_checkpoint_april13/checkpoints_3000000/` | EgoAllo模型路径 |
| `--smplh-npz-path` | `./data/smplh/neutral/model.npz` | SMPL-H模型路径 |

---

## 输出格式

### NPZ文件结构

推理结果保存为`.npz`文件，包含以下字段：

```python
{
    'Ts_world_cpf': [T, 7],           # 头部轨迹 (wxyz四元数 + xyz平移)
    'Ts_world_root': [T, 7],          # 身体根节点轨迹
    'body_quats': [1, T, 21, 4],      # 身体21个关节四元数
    'left_hand_quats': [1, T, 15, 4], # 左手15个关节四元数
    'right_hand_quats': [1, T, 15, 4],# 右手15个关节四元数
    'contacts': [1, T, 21],           # 接触标签
    'betas': [1, T, 16],              # SMPL-H形状参数
    'frame_nums': [T],                # 帧编号
    'timestamps_ns': [T],             # 时间戳（纳秒）
}
```

### 输出路径

```
ee4d_inference_output/
  └── indiana_cooking_23_5___0___513/
      └── 20260109-220807_0-64.npz
```

文件命名格式: `{日期时间}_{起始帧}-{结束帧}.npz`

---

## 数据格式转换

### EE4D → EgoAllo

EE4D数据使用**rotation_6d + translation [9]**格式  
EgoAllo需要**SE3 (quaternion + translation [7])**格式

转换过程（自动处理）:
1. rotation_6d → rotation_matrix (Gram-Schmidt正交化)
2. rotation_matrix → quaternion (Shepperd方法)
3. 调整地面高度 (floor_height参数)
4. 外推T0帧 (EgoAllo需要T+1个pose)

**验证精度**: 最大误差 ~1e-7

### 关键差异

| 特性 | UniEgoMotion (EE4D) | EgoAllo |
|------|---------------------|---------|
| 输入格式 | rotation_6d [9] | SE3 quaternion [7] |
| 输出格式 | SMPL-X (22 body + 15×2 hands) | SMPL-H (21 body + 15×2 hands) |
| 帧率 | 10 FPS | 30 FPS (原生) |
| 窗口长度 | 80 帧 | 32-128 帧 |
| 手部模型 | SMPL-X手型参数 | SMPL-H手型参数 |

---

## 常见问题

### Q1: CUDA不可用

```bash
# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"
# 应输出: 2.11.0.dev20260109+cu128

# 重新安装正确版本
pip uninstall torch torchvision torchaudio -y
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Q2: 推理速度慢

1. 关闭内部引导: `--guidance-inner False`
2. 使用`no_hands`模式
3. 减少轨迹长度: `--traj-length 32`

**参考速度** (RTX 5090):
- 64帧 + no_hands + post: ~7秒
- 64帧 + no_hands + inner+post: ~15秒

### Q3: 找不到序列

```bash
# 列出所有可用序列
python ee4d_adapter.py --list

# 输出前20个序列示例:
# indiana_cooking_23_5___0___513 (172帧)
# indiana_cooking_23_5___519___2472 (652帧)
# ...
```

### Q4: 内存不足

1. 减少轨迹长度: `--traj-length 32`
2. 减少采样数: `--num-samples 1`
3. 使用CPU模式（不推荐）

### Q5: 数据格式错误

检查ee_val.pt路径:
```python
# 默认路径
d:\Repository\UniEgoMotion\data\ee4d_motion_uniegomotion\uniegomotion\ee_val.pt
```

---

## 核心文件说明

### 推理脚本

- **`一键推理_EE4D.py`**: 批量推理脚本（推荐）
- **`ee4d_inference.py`**: 单个序列推理脚本
- **`ee4d_adapter.py`**: EE4D数据加载和格式转换

### 数据转换

- **`rotation_6d_to_matrix()`**: rotation_6d → 旋转矩阵
- **`matrix_to_quaternion_wxyz()`**: 旋转矩阵 → 四元数
- **`ee4d_aria_traj_to_egoallo()`**: EE4D格式 → EgoAllo格式
- **`MockAriaTraj`**: 模拟Aria轨迹对象

### 推理流程

```
load_ee4d_sequence()
  ↓
ee4d_aria_traj_to_egoallo()
  ↓
run_sampling_with_stitching()
  ↓
保存NPZ文件
```

---

## 下一步: 模型对比

### 1. 在同一序列上运行两个模型

**UniEgoMotion**:
```bash
cd d:\Repository\UniEgoMotion
conda activate uem
python 单次推理接口.py  # 配置sequence_name
```

**EgoAllo**:
```bash
cd d:\Repository\egoallo
conda activate egoallo
python ee4d_inference.py --sequence indiana_cooking_23_5___0___513
```

### 2. 格式对齐

- UniEgoMotion输出: `inference_output/<sequence>/infer_0.npz`
- EgoAllo输出: `ee4d_inference_output/<sequence>/YYYYMMDD-HHMMSS_0-64.npz`

### 3. 评估指标

- MPJPE (Mean Per Joint Position Error)
- PA-MPJPE (Procrustes Aligned MPJPE)
- 手部关节精度
- 滑步度量

### 4. 可视化对比

使用UniEgoMotion的可视化工具:
```bash
python 可视化推理结果.py
```

---

## 技术细节

### 数据精度验证

```python
# 测试旋转矩阵转换精度
python -c "
import torch
from ee4d_adapter import rotation_6d_to_matrix, matrix_to_quaternion_wxyz

rot_6d = torch.randn(100, 6)
rot_mat = rotation_6d_to_matrix(rot_6d)
quat = matrix_to_quaternion_wxyz(rot_mat)

# 验证正交性
identity = rot_mat @ rot_mat.transpose(-2, -1)
error = (identity - torch.eye(3)).abs().max()
print(f'最大误差: {error.item():.2e}')  # ~1e-7
"
```

### Gram-Schmidt正交化

确保rotation_6d转换后的旋转矩阵满足正交性:

```python
col1 = col1 / ||col1||
col2 = col2_raw - <col1, col2_raw> * col1
col2 = col2 / ||col2||
col3 = col1 × col2
```

### Shepperd四元数转换

根据旋转矩阵的trace选择最稳定的分支:
1. trace > 0: 使用w分量
2. m00最大: 使用x分量
3. m11最大: 使用y分量
4. m22最大: 使用z分量

---

## 性能基准

**测试环境**: RTX 5090, CUDA 12.8, PyTorch 2.11 nightly

| 配置 | 帧数 | 用时 | FPS |
|------|------|------|-----|
| no_hands + post | 64 | 7秒 | 9.1 |
| no_hands + inner + post | 64 | 15秒 | 4.3 |
| no_hands + post | 128 | 14秒 | 9.1 |

**扩散采样**: 30步 (~1秒)  
**JAX优化**: 6-7迭代 (~6秒)

---

## 参考资料

- **EgoAllo论文**: https://egoallo.github.io/
- **UniEgoMotion论文**: https://github.com/NetEase-GameAI/UniEgoMotion
- **EE4D数据集**: https://ego-exo4d-data.org/
- **SMPL-H模型**: https://mano.is.tue.mpg.de/

---

## 更新日志

- **2026-01-09**: 初始版本，成功在EE4D数据上运行EgoAllo推理
  - 创建数据适配器 (ee4d_adapter.py)
  - 实现推理脚本 (ee4d_inference.py)
  - 添加一键推理 (一键推理_EE4D.py)
  - 验证数据转换精度 (~1e-7)
  - 测试推理性能 (7秒/64帧)

---

**作者**: GitHub Copilot  
**更新**: 2026-01-09
