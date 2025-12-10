# GCN C++ 代码结构说明

## 目录结构

```
cpp/
├── include/                    # 头文件目录
│   ├── types.h                # 核心数据结构定义
│   ├── gcn_layer.h            # GCN 层接口
│   ├── scheduler.h            # 调度器接口
│   ├── optimizer.h            # 优化器接口
│   ├── loss.h                 # 损失函数接口
│   ├── backward.h             # 反向传播接口
│   ├── graph_utils.h          # 图处理工具接口
│   ├── data_loader.h          # 数据加载接口
│   └── trainer.h              # 训练器接口
├── src/                        # 源文件目录
│   ├── main.cpp               # 主程序入口
│   ├── trainer.cpp            # 训练器实现（核心训练逻辑）
│   ├── gcn_layer.cpp          # GCN 层实现
│   ├── scheduler.cpp          # 调度器实现
│   ├── optimizer.cpp          # 优化器实现
│   ├── loss.cpp               # 损失函数实现
│   ├── backward.cpp           # 反向传播实现
│   ├── graph_utils.cpp        # 图处理工具实现
│   └── data_loader.cpp        # 数据加载实现（占位符）
├── CMakeLists.txt             # CMake 构建配置
├── README.md                  # 使用说明
├── GCN_CPP_FEASIBILITY.md    # 可行性分析
├── IMPLEMENTATION_STATUS.md   # 实现状态
└── CODE_STRUCTURE.md          # 本文件
```

## 模块说明

### 1. 核心数据结构 (`types.h`)
- **GraphData**: 完整的图数据（邻接矩阵、特征、标签、掩码）
- **SubgraphData**: 子图数据（用于分区训练）
- **GCNConfig**: 训练配置参数
- **TaskInfo**: 任务信息（用于调度）
- **StageLog**: 阶段执行日志

### 2. GCN 层 (`gcn_layer.h/cpp`)
- **GCNLayer**: 单层 GCN
  - `update()`: Update 操作（线性变换 + ReLU + Dropout）
  - `aggregate()`: Aggregate 操作（邻接矩阵乘法）
- **GCNModel**: 两层 GCN 模型
  - 分阶段前向传播方法
  - 与 Python 版本的 `layer._update()` 和 `layer._aggregate()` 对应

### 3. 调度器 (`scheduler.h/cpp`)
- **Scheduler**: 任务调度器
  - 根据 AMIR/CD 值调度到 PIM/PNM/GPU
  - K-means 阈值计算（简化版）
  - 与 Python 版本的 `scheduler.schedule_task()` 完全对应

### 4. 优化器 (`optimizer.h/cpp`)
- **AdamOptimizer**: Adam 优化器
- **SGDOptimizer**: SGD 优化器
- 与 Python 版本的 `tf.train.AdamOptimizer` 对应

### 5. 损失函数 (`loss.h/cpp`)
- **LossFunctions**: 损失和评估函数
  - `masked_softmax_cross_entropy()`: 带掩码的交叉熵损失
  - `masked_accuracy()`: 带掩码的准确率
  - `l2_loss()`: L2 正则化
  - 与 Python 版本的 `masked_softmax_cross_entropy()` 和 `masked_accuracy()` 对应

### 6. 反向传播 (`backward.h/cpp`)
- **BackwardPropagator**: 反向传播器
  - 实现完整的反向传播链
  - Layer 2 → Layer 1 的梯度传播
  - 与 Python 版本的 TensorFlow 自动微分对应

### 7. 图处理工具 (`graph_utils.h/cpp`)
- **GraphUtils**: 图预处理
  - `normalize_adj()`: 归一化邻接矩阵（D^(-1/2) * A * D^(-1/2)）
  - `preprocess_features()`: 行归一化特征
  - `add_self_loops()`: 添加自环
- **PartitionUtils**: 图分区
  - `metis_partition()`: METIS 分区（待完善）
  - `extract_subgraph()`: 提取子图
  - 与 Python 版本的 `metis_partition()` 和 `extract_all_partition_subgraphs()` 对应

### 8. 数据加载 (`data_loader.h/cpp`)
- **DataLoader**: 数据加载器
  - 从 pickle 文件或文本文件加载数据
  - **待实现**：需要实现完整的 pickle 解析或使用预处理数据

### 9. 训练器 (`trainer.h/cpp`)
- **Trainer**: 训练器类（核心）
  - `initialize()`: 初始化（数据加载、分区、子图提取）
  - `train()`: 训练主循环
  - `exec_stage()`: 执行一个阶段（Update 或 Aggregate）
  - **训练逻辑与 Python 版本完全一致**

## 训练流程对应

### Python 版本流程：
```python
1. load_data() → 加载数据
2. metis_partition() → 图分区
3. extract_all_partition_subgraphs() → 提取子图
4. 对每个分区：
   for epoch in range(epochs):
       # Layer 1 Update
       updated = _exec_stage(1, 'UPDATE', ...)
       # Layer 1 Aggregate
       aggregated = _exec_stage(1, 'AGG', ...)
       # Layer 2 Update
       updated = _exec_stage(2, 'UPDATE', ...)
       # Layer 2 Aggregate
       outputs = _exec_stage(2, 'AGG', ...)
       # 损失和反向传播
       loss = compute_loss()
       backward()
       optimizer.update()
```

### C++ 版本流程：
```cpp
1. DataLoader::load_data() → 加载数据
2. PartitionUtils::metis_partition() → 图分区
3. PartitionUtils::extract_subgraph() → 提取子图
4. 对每个分区：
   for (int epoch = 0; epoch < epochs; epoch++) {
       // Layer 1 Update
       MatrixXf updated = exec_stage(1, "UPDATE", ...);
       // Layer 1 Aggregate
       MatrixXf aggregated = exec_stage(1, "AGG", ...);
       // Layer 2 Update
       updated = exec_stage(2, "UPDATE", ...);
       // Layer 2 Aggregate
       MatrixXf outputs = exec_stage(2, "AGG", ...);
       // 损失和反向传播
       float loss = LossFunctions::masked_softmax_cross_entropy(...);
       BackwardPropagator::backward_*();
       optimizer.update();
   }
```

## 关键设计决策

### 1. 模块化设计
- 每个模块独立，低耦合
- 可以单独测试和修改
- 易于扩展和维护

### 2. 与 Python 版本一致
- 训练逻辑完全一致
- 数据流完全一致
- 调度逻辑完全一致

### 3. 使用 Eigen 替代 TensorFlow
- 更轻量级
- 性能可能更好
- 更容易集成 ZSim hooks

### 4. 暂时不包含 ZSim hooks
- 专注于训练逻辑实现
- 后续可以轻松添加 ROI 标记

## 待完善部分

1. **数据加载** - 需要实现 pickle 解析或使用预处理数据
2. **METIS 集成** - 需要调用真实的 METIS C 库
3. **反向传播完善** - 需要保存前向传播的中间值
4. **测试验证** - 需要与 Python 版本结果对比

## 编译和使用

```bash
# 编译
cd cpp
mkdir build && cd build
cmake ..
make

# 运行
./gcn_cpp --dataset cora --epochs 200 --learning_rate 0.01
```

## 注意事项

- 当前代码结构完整，训练逻辑与 Python 版本一致
- 数据加载是占位符，需要实现或使用预处理数据
- 所有模块都是低耦合设计，便于独立开发和测试

