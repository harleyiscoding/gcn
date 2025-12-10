# GCN 与 ZSim/Ramulator 集成指南

本文档说明如何将 GCN 调度器与 ZSim/Ramulator 集成，实现当任务被分配到 PIM/PNM 时自动生成 trace 文件并进行仿真。

## 一、概述

### 1.1 功能说明

当 GCN 调度器将任务分配到 **PIM** 或 **PNM** 设备时，系统可以：
1. 使用 `zsim_hooks` 标记 ROI（Region of Interest）区域
2. 可选地调用 ZSim 生成完整的 trace 文件
3. 使用 Ramulator 对 trace 进行仿真分析

### 1.2 架构组件

- **`zsim_ramulator_executor.py`**: ZSim 和 Ramulator 执行器类
- **`task_scheduler_distributed.py`**: 集成了执行器的 GCN 训练脚本
- **`zsim_hooks_python.c`**: Python C 扩展，提供 ROI 标记功能

## 二、环境准备

### 2.1 编译 zsim_hooks Python 扩展

```bash
cd /home/wanyu/hlexperience/gcnex/gcn-master/gcn

# 编译 C 扩展
python3 setup.py build_ext --inplace

# 或者手动编译
gcc -shared -fPIC -I/usr/include/python3.6 \
    -I../../../ramulator-pim-master/zsim-ramulator/misc/hooks \
    zsim_hooks_python.c -o zsim_hooks_python.so
```

### 2.2 Docker 环境（可选）

如果使用 Docker 模式，确保 Docker 镜像已构建：

```bash
cd /home/wanyu/hlexperience/ramulator-pim-master
docker build -t ramulator-pim:latest .
```

## 三、使用方法

### 3.1 基本使用（仅 ROI 标记）

最简单的使用方式是只启用 ROI 标记，不生成完整 trace：

```bash
cd /home/wanyu/hlexperience/gcnex/gcn-master

python3 -m gcn.task_scheduler_distributed \
    --dataset cora \
    --epochs 10 \
    --num_parts 4
```

**注意**：这种方式需要在 ZSim 环境下运行 Python 脚本才能生成 trace。

### 3.2 启用 ZSim/Ramulator 集成

要启用完整的 ZSim/Ramulator 集成：

```bash
python3 -m gcn.task_scheduler_distributed \
    --dataset cora \
    --epochs 10 \
    --num_parts 4 \
    --enable_zsim_ramulator \
    --use_docker  # 如果使用 Docker 模式
```

**参数说明**：
- `--enable_zsim_ramulator`: 启用 ZSim/Ramulator 集成
- `--use_docker`: 使用 Docker 容器运行 ZSim/Ramulator（推荐）
- `--zsim_output_dir`: 指定 trace 和统计文件输出目录（可选）

### 3.3 本地模式（不使用 Docker）

如果 ZSim 和 Ramulator 已安装在本地：

```bash
python3 -m gcn.task_scheduler_distributed \
    --dataset cora \
    --epochs 10 \
    --num_parts 4 \
    --enable_zsim_ramulator
    # 不设置 --use_docker
```

## 四、工作流程

### 4.1 调度流程

1. **任务调度**：调度器根据 AMIR/CD 值将任务分配到 GPU/PIM/PNM
2. **ROI 标记**：如果任务分配到 PIM/PNM 且 `zsim_hooks` 可用，自动标记 ROI
3. **执行计算**：在 TensorFlow session 中执行计算
4. **Trace 生成**（可选）：如果启用了执行器，可以异步生成 trace

### 4.2 ROI 标记

当任务被分配到 PIM/PNM 时，代码会自动在计算前后添加 ROI 标记：

```python
if device in ('PIM', 'PNM') and executor is not None:
    if ZSIM_HOOKS_AVAILABLE:
        zsim_hooks.zsim_roi_begin()  # 开始 ROI
    # ... 执行计算 ...
    if ZSIM_HOOKS_AVAILABLE:
        zsim_hooks.zsim_roi_end()    # 结束 ROI
```

### 4.3 Trace 生成（未来扩展）

当前实现主要提供 ROI 标记功能。完整的 trace 生成需要：
1. 在 ZSim 环境下运行 Python 脚本
2. 使用 ZSim 配置文件指定要插桩的进程
3. ZSim 会自动生成 trace 文件

## 五、输出文件

### 5.1 Trace 文件

如果启用了执行器，trace 文件会保存在：
```
gcn/results/zsim_ramulator_traces/
├── gcn_PIM_AGG_L1_P0_E0_*.out
├── gcn_PNM_UPDATE_L1_P0_E0_*.out
└── ...
```

### 5.2 Ramulator 统计文件

Ramulator 仿真统计文件：
```
gcn/results/zsim_ramulator_traces/
├── ramulator_stats_gcn_PIM_AGG_L1_P0_E0_*.stats
└── ...
```

## 六、配置说明

### 6.1 ZSim 配置

执行器会自动生成 ZSim 配置文件，基于模板 `zsim-ramulator/tests/host.cfg`。

关键配置项：
- `cores`: CPU 核心数（默认 1）
- `outFile`: trace 输出文件名
- `splitTrace`: 是否生成 per-core trace（默认 false）
- `process0.command`: Python 脚本命令

### 6.2 Ramulator 配置

使用 Ramulator 默认配置 `ramulator/Configs/host.cfg`。

关键参数：
- `--mode`: 仿真模式（`cpu` 用于 PNM，`pim` 用于 PIM）
- `--number-cores`: 核心数（默认 1）
- `--trace-format`: trace 格式（`zsim`）

## 七、故障排除

### 7.1 zsim_hooks 导入失败

**错误**：`ImportError: No module named 'zsim_hooks_python'`

**解决**：
1. 确保已编译 `zsim_hooks_python.so`
2. 确保 `.so` 文件在 Python 路径中
3. 检查 C 扩展的依赖库是否正确链接

### 7.2 ZSim 执行失败

**错误**：`ZSim execution failed`

**可能原因**：
1. ZSim 路径不正确
2. Docker 容器未启动或镜像不存在
3. 配置文件格式错误

**解决**：
1. 检查 `zsim_path` 和 `ramulator_path` 是否正确
2. 如果使用 Docker，确保镜像已构建：`docker images | grep ramulator-pim`
3. 检查生成的配置文件格式

### 7.3 Ramulator 执行失败

**错误**：`Ramulator failed: Bad trace file`

**可能原因**：
1. Trace 文件不存在或为空
2. Trace 格式不匹配

**解决**：
1. 检查 trace 文件是否成功生成：`ls -lh results/zsim_ramulator_traces/*.out`
2. 检查 trace 文件大小（不应为 0）
3. 验证 trace 格式是否正确

## 八、高级用法

### 8.1 自定义执行器配置

```python
from zsim_ramulator_executor import ZSimRamulatorExecutor

executor = ZSimRamulatorExecutor(
    zsim_path="/custom/path/to/zsim",
    ramulator_path="/custom/path/to/ramulator",
    zsim_config_template="/custom/path/to/config.cfg",
    output_dir="/custom/output/dir",
    use_docker=False
)
```

### 8.2 批量生成 Trace

可以修改代码，在训练完成后批量生成所有任务的 trace：

```python
# 收集所有需要 trace 的任务
trace_tasks = []
for log in stage_device_log:
    if log[6] in ('PIM', 'PNM'):  # device
        trace_tasks.append(log)

# 批量执行
for task in trace_tasks:
    executor.execute_task(...)
```

## 九、性能考虑

### 9.1 ROI 标记开销

ROI 标记的开销很小（只是函数调用），不会显著影响训练性能。

### 9.2 Trace 生成开销

如果启用完整的 trace 生成，会显著增加执行时间：
- ZSim 插桩会降低执行速度（通常 10-100x）
- Ramulator 仿真时间取决于 trace 大小

**建议**：
- 开发/调试时：启用完整 trace 生成
- 生产训练时：仅启用 ROI 标记，后续批量生成 trace

## 十、示例输出

### 10.1 调度日志

```
[调度器] Layer 1 Update: 值=2.3456, 分配到 PIM
[完成] Layer 1 Update 在 PIM 上完成 (Partition 0)
INFO: Layer 1 Update executed on PIM (ROI marked for ZSim)

[调度器] Layer 1 Aggregate: 值=0.1234, 分配到 PNM
[完成] Layer 1 Aggregate 在 PNM 上完成 (Partition 0)
INFO: Layer 1 Aggregate executed on PNM (ROI marked for ZSim)
```

### 10.2 执行器日志

```
INFO: ZSimRamulatorExecutor initialized:
  - Use Docker: True
  - ZSim path: /workspace/zsim-ramulator/build/opt/zsim
  - Ramulator path: /workspace/ramulator/ramulator
  - Output dir: /path/to/output
INFO: Generated ZSim config: /path/to/zsim_config_L1_U_P0_E0.cfg
INFO: ZSim trace generated: /path/to/gcn_PIM_UPDATE_L1_P0_E0.out
INFO: Ramulator stats generated: /path/to/ramulator_stats_*.stats
```

## 十一、下一步

1. **完善 trace 生成**：实现完整的独立进程 trace 生成
2. **性能分析**：集成 Ramulator 统计结果到调度决策
3. **可视化**：创建 trace 和统计结果的可视化工具
4. **自动化测试**：添加集成测试确保功能正常

---

**文档版本**：v1.0  
**最后更新**：2024  
**维护者**：项目团队

