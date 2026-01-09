"""
快速测试 - 验证对比流程是否工作
只对比1个序列，用于调试
"""

import subprocess
from pathlib import Path
import time

# 测试配置
TEST_CONFIG = {
    "sequence": "indiana_cooking_23_5___0___513",
    "uniegomotion_root": r"d:\Repository\UniEgoMotion",
    "egoallo_root": r"d:\Repository\egoallo",
    "uem_env": "uem",
    "egoallo_env": "egoallo",
    "output_root": "./test_comparison_output"
}

def test_conda_run():
    """测试conda run命令是否可用"""
    print("测试conda run命令...")
    
    try:
        result = subprocess.run(
            "conda run -n base python --version",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✓ conda run可用")
            print(f"  输出: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ conda run失败")
            print(f"  错误: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"✗ conda run异常: {e}")
        return False


def test_environments():
    """测试两个conda环境是否存在"""
    print("\n测试conda环境...")
    
    for env_name in [TEST_CONFIG['uem_env'], TEST_CONFIG['egoallo_env']]:
        try:
            result = subprocess.run(
                f"conda run -n {env_name} python --version",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"✓ 环境 '{env_name}' 存在")
                print(f"  Python: {result.stdout.strip()}")
            else:
                print(f"✗ 环境 '{env_name}' 不可用")
                return False
        
        except Exception as e:
            print(f"✗ 环境 '{env_name}' 测试失败: {e}")
            return False
    
    return True


def test_egoallo_inference():
    """测试EgoAllo推理"""
    print("\n测试EgoAllo推理...")
    
    output_dir = Path(TEST_CONFIG['output_root']) / "egoallo" / TEST_CONFIG['sequence']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    command = (
        f"conda run -n {TEST_CONFIG['egoallo_env']} "
        f"python ee4d_inference.py "
        f"--sequence {TEST_CONFIG['sequence']} "
        f"--traj-length 64 "
        f"--output-dir {output_dir}"
    )
    
    print(f"命令: {command}")
    print(f"工作目录: {TEST_CONFIG['egoallo_root']}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=TEST_CONFIG['egoallo_root'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ EgoAllo推理成功 (用时: {elapsed:.2f}秒)")
            
            # 检查输出文件
            npz_files = list(output_dir.glob("*.npz"))
            if npz_files:
                print(f"  输出文件: {npz_files[0].name}")
            else:
                print(f"  警告: 未找到输出NPZ文件")
            
            return True
        else:
            print(f"✗ EgoAllo推理失败")
            print(f"  错误: {result.stderr[:500]}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"✗ EgoAllo推理超时（>10分钟）")
        return False
    
    except Exception as e:
        print(f"✗ EgoAllo推理异常: {e}")
        return False


def test_uniegomotion_inference():
    """测试UniEgoMotion推理"""
    print("\n测试UniEgoMotion推理...")
    
    output_dir = Path(TEST_CONFIG['output_root']) / "uniegomotion" / TEST_CONFIG['sequence']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    command = (
        f"conda run -n {TEST_CONFIG['uem_env']} "
        f"python uem_inference_interface.py "
        f"--sequence {TEST_CONFIG['sequence']} "
        f"--traj-length 64 "
        f"--output-dir {output_dir}"
    )
    
    print(f"命令: {command}")
    print(f"工作目录: {TEST_CONFIG['uniegomotion_root']}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=TEST_CONFIG['uniegomotion_root'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ UniEgoMotion推理成功 (用时: {elapsed:.2f}秒)")
            
            # 检查输出文件
            output_file = output_dir / "infer_0.npz"
            if output_file.exists():
                print(f"  输出文件: {output_file.name}")
            else:
                print(f"  警告: 未找到输出文件 {output_file}")
            
            return True
        else:
            print(f"✗ UniEgoMotion推理失败")
            print(f"  错误: {result.stderr[:500]}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"✗ UniEgoMotion推理超时（>10分钟）")
        return False
    
    except Exception as e:
        print(f"✗ UniEgoMotion推理异常: {e}")
        return False


def main():
    """主测试流程"""
    print("="*80)
    print(" 一键对比流程测试")
    print("="*80)
    
    # 测试1: conda run命令
    if not test_conda_run():
        print("\n❌ conda run命令不可用，无法继续测试")
        return
    
    # 测试2: conda环境
    if not test_environments():
        print("\n❌ conda环境检查失败")
        return
    
    # 测试3: EgoAllo推理
    egoallo_ok = test_egoallo_inference()
    
    # 测试4: UniEgoMotion推理
    uniegomotion_ok = test_uniegomotion_inference()
    
    # 汇总
    print("\n" + "="*80)
    print(" 测试结果汇总")
    print("="*80)
    
    print(f"\n✓ conda run: 可用")
    print(f"✓ conda环境: 正常")
    print(f"{'✓' if egoallo_ok else '✗'} EgoAllo推理: {'成功' if egoallo_ok else '失败'}")
    print(f"{'✓' if uniegomotion_ok else '✗'} UniEgoMotion推理: {'成功' if uniegomotion_ok else '失败'}")
    
    if egoallo_ok and uniegomotion_ok:
        print("\n🎉 所有测试通过！可以运行完整的一键对比脚本了")
        print(f"\n运行命令: python 一键对比_两模型.py")
    else:
        print("\n⚠️ 部分测试失败，请检查上述错误信息")
    
    print("="*80)


if __name__ == '__main__':
    main()
