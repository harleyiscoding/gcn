#!/usr/bin/env python3
"""
测试 ZSim/Ramulator 集成

用于验证 zsim_hooks 和执行器是否正常工作
"""

import os
import sys

# 测试 zsim_hooks 导入
print("=" * 60)
print("测试 1: zsim_hooks 导入")
print("=" * 60)
try:
    import zsim_hooks_python as zsim_hooks
    print("✓ zsim_hooks_python 导入成功")
    
    # 测试函数调用
    try:
        zsim_hooks.zsim_roi_begin()
        print("✓ zsim_roi_begin() 调用成功")
        zsim_hooks.zsim_roi_end()
        print("✓ zsim_roi_end() 调用成功")
    except Exception as e:
        print(f"✗ zsim_hooks 函数调用失败: {e}")
        print("  注意：这可能是正常的，如果不在 ZSim 环境下运行")
    
except ImportError as e:
    print(f"✗ zsim_hooks_python 导入失败: {e}")
    print("  提示：需要编译 zsim_hooks_python.c")
    print("  命令：python3 setup.py build_ext --inplace")

# 测试执行器导入
print("\n" + "=" * 60)
print("测试 2: ZSimRamulatorExecutor 导入")
print("=" * 60)
try:
    from zsim_ramulator_executor import ZSimRamulatorExecutor
    print("✓ ZSimRamulatorExecutor 导入成功")
    
    # 测试执行器初始化
    try:
        executor = ZSimRamulatorExecutor(
            use_docker=False,  # 先测试本地模式
            output_dir="/tmp/zsim_test"
        )
        print("✓ ZSimRamulatorExecutor 初始化成功")
        print(f"  - ZSim path: {executor.zsim_path}")
        print(f"  - Ramulator path: {executor.ramulator_path}")
        print(f"  - Output dir: {executor.output_dir}")
        
        # 检查路径是否存在
        if os.path.exists(executor.zsim_path):
            print(f"  ✓ ZSim 可执行文件存在")
        else:
            print(f"  ✗ ZSim 可执行文件不存在: {executor.zsim_path}")
        
        if os.path.exists(executor.ramulator_path):
            print(f"  ✓ Ramulator 可执行文件存在")
        else:
            print(f"  ✗ Ramulator 可执行文件不存在: {executor.ramulator_path}")
            
    except Exception as e:
        print(f"✗ ZSimRamulatorExecutor 初始化失败: {e}")
        
except ImportError as e:
    print(f"✗ ZSimRamulatorExecutor 导入失败: {e}")

# 测试调度器集成
print("\n" + "=" * 60)
print("测试 3: 调度器集成检查")
print("=" * 60)
try:
    # 检查 task_scheduler_distributed.py 中的集成代码
    scheduler_file = os.path.join(os.path.dirname(__file__), "task_scheduler_distributed.py")
    if os.path.exists(scheduler_file):
        with open(scheduler_file, 'r') as f:
            content = f.read()
        
        checks = [
            ("ZSimRamulatorExecutor", "执行器导入"),
            ("enable_zsim_ramulator", "FLAGS 定义"),
            ("zsim_hooks.zsim_roi_begin", "ROI 标记"),
            ("device in ('PIM', 'PNM')", "设备检查")
        ]
        
        for keyword, description in checks:
            if keyword in content:
                print(f"✓ {description}: 已集成")
            else:
                print(f"✗ {description}: 未找到")
    else:
        print(f"✗ 找不到 task_scheduler_distributed.py")
        
except Exception as e:
    print(f"✗ 检查失败: {e}")

# 总结
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)
print("\n如果所有测试通过，可以运行：")
print("  python3 -m gcn.task_scheduler_distributed \\")
print("    --dataset cora \\")
print("    --epochs 5 \\")
print("    --num_parts 2 \\")
print("    --enable_zsim_ramulator")
print("\n注意：")
print("  1. 如果 zsim_hooks 不可用，ROI 标记会被跳过，但不影响训练")
print("  2. 完整的 trace 生成需要在 ZSim 环境下运行整个训练脚本")
print("  3. 使用 --use_docker 可以自动使用 Docker 容器运行 ZSim/Ramulator")

