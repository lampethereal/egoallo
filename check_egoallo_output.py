"""查看EgoAllo推理输出的格式"""
import numpy as np
from pathlib import Path

# 加载最新的输出
output_file = Path("egoallo_example_trajectories/coffeemachine/egoallo_outputs/20260109-220017_0-64.npz")

print(f"加载输出文件: {output_file}\n")

data = np.load(output_file)

print("=" * 70)
print("EgoAllo输出格式")
print("=" * 70)

for key in data.files:
    value = data[key]
    print(f"\n{key}:")
    print(f"  形状: {value.shape}")
    print(f"  类型: {value.dtype}")
    if len(value.shape) <= 2 and value.shape[0] <= 5:
        print(f"  数据:\n{value}")
    else:
        print(f"  前3帧: {value[:3]}")

print("\n" + "=" * 70)
print("数据说明:")
print("=" * 70)
print("""
- Ts_world_cpf: [T, 7] - 头部轨迹 (wxyz四元数 + xyz平移)
- Ts_world_root: [T, 7] - 身体根节点轨迹
- body_quats: [T, 21, 4] - 身体21个关节的四元数
- left_hand_quats: [T, 15, 4] - 左手15个关节
- right_hand_quats: [T, 15, 4] - 右手15个关节
- contacts: [T, 8] - 接触标签
- betas: [10] - SMPL-H形状参数
- frame_nums: [T] - 帧编号
- timestamps_ns: [T] - 时间戳（纳秒）
""")

print("✓ EgoAllo推理成功!")
print(f"✓ 生成了{data['body_quats'].shape[0]}帧的SMPL-H动作")
