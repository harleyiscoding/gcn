# GCN 调度器调用链总结

## 快速参考：从入口到执行的完整调用链

### 入口点
```
python3 -m gcn.task_scheduler_distributed --enable_zsim_ramulator
    │
    └─> task_scheduler_distributed.py (第 298 行)
```

### 完整调用链

```
【1. 初始化阶段】
├─> load_data(FLAGS.dataset)
│   └─> 返回: adj, features, y_train, y_val, y_test, masks
│
├─> metis_partition(adj, num_parts)
│   └─> 返回: part_labels (分区标签)
│
├─> extract_all_partition_subgraphs(...)
│   └─> 返回: subgraph_list (子图列表)
│
├─> 读取 memory_flops_epochs.txt
│   └─> 计算 AMIR 和 CD
│   └─> 生成 tasks_info 列表
│
├─> scheduler = Scheduler()  # 第 418 行
│   └─> 初始化阈值和历史记录
│
└─> executor = ZSimRamulatorExecutor(...)  # 第 425-428 行
    └─> 如果 --enable_zsim_ramulator 启用
    └─> 初始化执行器（本地或 Docker 模式）

【2. 分区循环】
└─> for part_id, subgraph in enumerate(subgraph_list):  # 第 430 行
    │
    ├─> tf.reset_default_graph()  # 为每个分区重置图
    ├─> 解压缩子图数据
    ├─> 构建模型和 Session
    │
    └─> for epoch in range(FLAGS.epochs):  # 第 514 行
        │
        └─> 【每个 epoch 执行 4 个阶段】

【3. 阶段执行（以 Layer 1 Update 为例）】
└─> 第 525-554 行
    │
    ├─> 【步骤 1】获取任务信息
    │   └─> idx = epoch * 4 + 0
    │   └─> task = tasks_info[idx]
    │       └─> {'layer': 1, 'phase': 'UPDATE', 'value': cd1}
    │
    ├─> 【步骤 2】调度决策
    │   └─> device = scheduler.schedule_task('UPDATE', cd1)  # 第 529 行
    │       │
    │       └─> Scheduler.schedule_task() 内部流程:  # 第 75-93 行
    │           ├─> 记录历史值: self.history_cd.append(cd1)
    │           ├─> 更新阈值: self.update_thresholds()  # 第 63-73 行
    │           │   └─> 如果历史记录 >= 20，使用 KMeans 聚类
    │           │
    │           └─> 决策逻辑:  # 第 89-93 行
    │               if value > self.cd_threshold:
    │                   return 'GPU'
    │               else:
    │                   return 'PNM'
    │
    ├─> 【步骤 3】记录日志
    │   └─> stage_device_log.append(...)  # 第 531 行
    │
    └─> 【步骤 4】执行计算
        │
        ├─> 情况 A: device == 'GPU'  # 第 548-550 行
        │   └─> updated = sess.run([model.layers[0]._update(...)])
        │
        ├─> 情况 B: device in ('PIM', 'PNM') and executor is not None  # 第 538-547 行
        │   ├─> zsim_hooks.zsim_roi_begin()  # 第 541 行
        │   │   └─> 调用 zsim_hooks_python.c (编译为 .so)
        │   │       └─> 调用 zsim-ramulator 的 zsim_roi_begin()
        │   │
        │   ├─> updated = sess.run([model.layers[0]._update(...)])  # 第 543 行
        │   │   └─> TensorFlow 执行计算
        │   │
        │   └─> zsim_hooks.zsim_roi_end()  # 第 545 行
        │
        └─> 情况 C: 默认执行  # 第 551-553 行
            └─> updated = sess.run([model.layers[0]._update(...)])
```

## 关键接口调用位置

### 1. Scheduler.schedule_task()
```python
# 文件: task_scheduler_distributed.py
# 行号: 75-93

调用位置:
  - 第 529 行: device = scheduler.schedule_task('UPDATE', cd1)
  - 第 533 行: device = scheduler.schedule_task('UPDATE', 1.0)  # 默认值
  - 第 560 行: device = scheduler.schedule_task('AGG', amir1)
  - 第 564 行: device = scheduler.schedule_task('AGG', 1.0)
  - 第 587 行: device = scheduler.schedule_task('UPDATE', cd2)
  - 第 591 行: device = scheduler.schedule_task('UPDATE', 1.0)
  - 第 614 行: device = scheduler.schedule_task('AGG', amir2)
  - 第 618 行: device = scheduler.schedule_task('AGG', 1.0)

输入:
  - phase: 'AGG' 或 'UPDATE'
  - value: AMIR 值（AGG）或 CD 值（UPDATE）

输出:
  - 'GPU', 'PIM', 或 'PNM'
```

### 2. zsim_hooks.zsim_roi_begin() / zsim_roi_end()
```python
# 文件: zsim_hooks_python.c (编译为 zsim_hooks_python.so)
# 调用位置: task_scheduler_distributed.py

调用位置:
  - 第 541 行: zsim_hooks.zsim_roi_begin()  # Layer 1 Update
  - 第 545 行: zsim_hooks.zsim_roi_end()
  - 第 572 行: zsim_hooks.zsim_roi_begin()  # Layer 1 Aggregate
  - 第 575 行: zsim_hooks.zsim_roi_end()
  - 第 599 行: zsim_hooks.zsim_roi_begin()  # Layer 2 Update
  - 第 602 行: zsim_hooks.zsim_roi_end()
  - 第 626 行: zsim_hooks.zsim_roi_begin()  # Layer 2 Aggregate
  - 第 629 行: zsim_hooks.zsim_roi_end()

条件:
  - device in ('PIM', 'PNM')
  - executor is not None
  - ZSIM_HOOKS_AVAILABLE == True
```

### 3. ZSimRamulatorExecutor (当前主要用于框架)
```python
# 文件: zsim_ramulator_executor.py
# 初始化位置: task_scheduler_distributed.py 第 425-428 行

当前状态:
  - 已初始化，但未在训练循环中直接调用
  - 主要用于未来扩展（完整 trace 生成）

未来调用位置（待实现）:
  - executor.execute_task(...)  # 在训练完成后批量生成 trace
```

## 数据流

### tasks_info 生成
```
memory_flops_epochs.txt
    │
    └─> 读取文件 (第 386-416 行)
        │
        ├─> 计算 AMIR 和 CD
        │   ├─> AMIR = agg_mem / update_flops
        │   └─> CD = update_flops / agg_mem
        │
        └─> 生成 tasks_info 列表
            │
            └─> 每个 epoch 4 个任务:
                ├─> tasks_info[epoch*4 + 0] = {'layer': 1, 'phase': 'UPDATE', 'value': cd1}
                ├─> tasks_info[epoch*4 + 1] = {'layer': 1, 'phase': 'AGG', 'value': amir1}
                ├─> tasks_info[epoch*4 + 2] = {'layer': 2, 'phase': 'UPDATE', 'value': cd2}
                └─> tasks_info[epoch*4 + 3] = {'layer': 2, 'phase': 'AGG', 'value': amir2}
```

### 调度决策流程
```
tasks_info[idx]
    │
    └─> scheduler.schedule_task(phase, value)
        │
        ├─> 记录历史值
        │   ├─> if phase == 'AGG': self.history_amir.append(value)
        │   └─> else: self.history_cd.append(value)
        │
        ├─> 更新阈值
        │   └─> self.update_thresholds()
        │       ├─> 如果 history_amir >= 20: KMeans 聚类更新 amir_threshold
        │       └─> 如果 history_cd >= 20: KMeans 聚类更新 cd_threshold
        │
        └─> 决策
            ├─> if phase == 'AGG':
            │   ├─> if value > self.amir_threshold: return 'PIM'
            │   └─> else: return 'PNM'
            │
            └─> elif phase == 'UPDATE':
                ├─> if value > self.cd_threshold: return 'GPU'
                └─> else: return 'PNM'
```

## 执行路径选择

```
device = scheduler.schedule_task(...)
    │
    ├─> 返回 'GPU'
    │   └─> 直接执行: sess.run([model.layers[...]._update(...)])
    │
    ├─> 返回 'PIM' 或 'PNM'
    │   └─> 检查: executor is not None?
    │       │
    │       ├─> 是 → 检查: ZSIM_HOOKS_AVAILABLE?
    │       │   │
    │       │   ├─> 是 → ROI 标记 + 执行计算
    │       │   │   ├─> zsim_hooks.zsim_roi_begin()
    │       │   │   ├─> sess.run([...])
    │       │   │   └─> zsim_hooks.zsim_roi_end()
    │       │   │
    │       │   └─> 否 → 仅执行计算（无 ROI 标记）
    │       │       └─> sess.run([...])
    │       │
    │       └─> 否 → 默认执行
    │           └─> sess.run([...])
    │
    └─> 其他情况
        └─> 默认执行: sess.run([...])
```

## 关键代码位置速查表

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 主入口 | `task_scheduler_distributed.py` | 298 | `if __name__ == "__main__"` |
| 调度器类定义 | `task_scheduler_distributed.py` | 51-93 | `class Scheduler` |
| 调度器初始化 | `task_scheduler_distributed.py` | 418 | `scheduler = Scheduler()` |
| 执行器初始化 | `task_scheduler_distributed.py` | 425-428 | `executor = ZSimRamulatorExecutor(...)` |
| 分区循环 | `task_scheduler_distributed.py` | 430 | `for part_id, subgraph in enumerate(...)` |
| Epoch 循环 | `task_scheduler_distributed.py` | 514 | `for epoch in range(FLAGS.epochs)` |
| Layer 1 Update | `task_scheduler_distributed.py` | 525-554 | 第一个阶段 |
| Layer 1 Aggregate | `task_scheduler_distributed.py` | 556-581 | 第二个阶段 |
| Layer 2 Update | `task_scheduler_distributed.py` | 583-608 | 第三个阶段 |
| Layer 2 Aggregate | `task_scheduler_distributed.py` | 610-635 | 第四个阶段 |
| ROI 标记开始 | `task_scheduler_distributed.py` | 541, 572, 599, 626 | `zsim_hooks.zsim_roi_begin()` |
| ROI 标记结束 | `task_scheduler_distributed.py` | 545, 575, 602, 629 | `zsim_hooks.zsim_roi_end()` |
| zsim_hooks 定义 | `zsim_hooks_python.c` | 1-48 | C 扩展源码 |
| 执行器类定义 | `zsim_ramulator_executor.py` | 21-564 | `class ZSimRamulatorExecutor` |

## 调试建议

### 1. 检查调度决策
```python
# 在 scheduler.schedule_task() 中添加日志
print(f"[DEBUG] schedule_task: phase={phase}, value={value}, threshold={self.cd_threshold}")
```

### 2. 检查 ROI 标记
```python
# 在 ROI 标记前后添加日志
print(f"[DEBUG] Before ROI: device={device}, executor={executor}, hooks_available={ZSIM_HOOKS_AVAILABLE}")
zsim_hooks.zsim_roi_begin()
print(f"[DEBUG] ROI started")
# ... 执行计算 ...
zsim_hooks.zsim_roi_end()
print(f"[DEBUG] ROI ended")
```

### 3. 检查执行器状态
```python
# 在初始化后检查
if executor:
    print(f"[DEBUG] Executor initialized: {executor.zsim_path}, {executor.ramulator_path}")
else:
    print(f"[DEBUG] Executor not initialized")
```

---

**文档版本**: v1.0  
**最后更新**: 2024

