"""
快速对比测试 - 简化版
直接使用两个项目原本的一键推理脚本
"""

import subprocess
import time
from pathlib import Path

# 配置
SEQUENCE = "indiana_cooking_23_5___0___513"
UEM_ROOT = r"d:\Repository\UniEgoMotion"
EGOALLO_ROOT = r"d:\Repository\egoallo"

print("="*80)
print("快速对比测试")
print("="*80)

# 1. 测试EgoAllo推理
print("\n[1/2] 测试EgoAllo推理...")
start = time.time()
result_egoallo = subprocess.run(
    f"conda run -n egoallo python ee4d_inference.py --sequence {SEQUENCE} --traj-length 64",
    shell=True,
    cwd=EGOALLO_ROOT,
    capture_output=True,
    text=True
)
egoallo_time = time.time() - start

if result_egoallo.returncode == 0:
    print(f"✓ EgoAllo成功 ({egoallo_time:.2f}秒)")
else:
    print(f"✗ EgoAllo失败")
    print(result_egoallo.stderr[-500:])

# 2. 测试UniEgoMotion推理 - 使用原本的一键推理脚本
print("\n[2/2] 测试UniEgoMotion推理...")

# 修改一键推理.py的配置
uem_script = Path(UEM_ROOT) / "一键推理.py"
with open(uem_script, 'r', encoding='utf-8') as f:
    content = f.read()

# 备份
uem_backup = uem_script.with_suffix('.py.bak')
with open(uem_backup, 'w', encoding='utf-8') as f:
    f.write(content)

# 修改SPECIFIC_SEQ
import re
content = re.sub(
    r'"SPECIFIC_SEQ": ".*?"',
    f'"SPECIFIC_SEQ": "{SEQUENCE}"',
    content
)

# 写回
with open(uem_script, 'w', encoding='utf-8') as f:
    f.write(content)

# 运行
start = time.time()
result_uem = subprocess.run(
    f"conda run -n uem python 一键推理.py",
    shell=True,
    cwd=UEM_ROOT,
    capture_output=True,
    text=True
)
uem_time = time.time() - start

# 恢复
with open(uem_backup, 'r', encoding='utf-8') as f:
    content = f.read()
with open(uem_script, 'w', encoding='utf-8') as f:
    f.write(content)

if result_uem.returncode == 0:
    print(f"✓ UniEgoMotion成功 ({uem_time:.2f}秒)")
else:
    print(f"✗ UniEgoMotion失败")
    print(result_uem.stderr[-500:])

# 汇总
print("\n" + "="*80)
print("测试结果")
print("="*80)
print(f"EgoAllo: {'✓' if result_egoallo.returncode == 0 else '✗'} ({egoallo_time:.2f}秒)")
print(f"UniEgoMotion: {'✓' if result_uem.returncode == 0 else '✗'} ({uem_time:.2f}秒)")
print("="*80)
