# GCN C++ 实现状态

## 已完成模块 ✅

### 1. 核心数据结构 (`include/types.h`)
- ✅ GraphData - 图数据结构
- ✅ SubgraphData - 子图数据结构
- ✅ GCNConfig - 配置结构
- ✅ TaskInfo, StageLog, TrainingStats - 辅助数据结构

### 2. GCN 层实现 (`include/gcn_layer.h`, `src/gcn_layer.cpp`)
- ✅ GCNLayer - 单层 GCN 实现
  - ✅ Update 操作（线性变换 + ReLU + Dropout）
  - ✅ Aggregate 操作（邻接矩阵乘法）
- ✅ GCNModel - 两层 GCN 模型
  - ✅ 分阶段前向传播（layer1_update, layer1_aggregate, layer2_update, layer2_aggregate）

### 3. 调度器 (`include/scheduler.h`, `src/scheduler.cpp`)
- ✅ Scheduler - 任务调度器
  - ✅ 根据 AMIR/CD 值调度任务
  - ✅ K-means 阈值计算（简化版）

### 4. 优化器 (`include/optimizer.h`, `src/optimizer.cpp`)
- ✅ AdamOptimizer - Adam 优化器实现
- ✅ SGDOptimizer - SGD 优化器实现

### 5. 损失函数 (`include/loss.h`, `src/loss.cpp`)
- ✅ masked_softmax_cross_entropy - 带掩码的交叉熵损失
- ✅ masked_accuracy - 带掩码的准确率计算
- ✅ l2_loss - L2 正则化损失

### 6. 反向传播 (`include/backward.h`, `src/backward.cpp`)
- ✅ BackwardPropagator - 反向传播器
  - ✅ Layer 2 Aggregate 反向传播
  - ✅ Layer 2 Update 反向传播
  - ✅ Layer 1 Aggregate 反向传播
  - ✅ Layer 1 Update 反向传播

### 7. 图处理工具 (`include/graph_utils.h`, `src/graph_utils.cpp`)
- ✅ GraphUtils - 图预处理
  - ✅ normalize_adj - 归一化邻接矩阵
  - ✅ preprocess_features - 预处理特征
  - ✅ sparse_to_tuple / tuple_to_sparse - 稀疏矩阵转换
  - ✅ add_self_loops - 添加自环
- ✅ PartitionUtils - 图分区工具
  - ✅ metis_partition - METIS 分区（占位符）
  - ✅ get_partition_masks - 获取分区掩码
  - ✅ extract_subgraph - 提取子图

### 8. 数据加载 (`include/data_loader.h`, `src/data_loader.cpp`)
- ⚠️ DataLoader - 数据加载器（占位符，需要实现）

### 9. 训练器 (`include/trainer.h`, `src/trainer.cpp`)
- ✅ Trainer - 训练器类
  - ✅ 初始化（数据加载、分区、子图提取）
  - ✅ 训练循环（完全按照 Python 版本逻辑）
  - ✅ 阶段执行（exec_stage）
  - ✅ 评估函数
  - ✅ 统计信息输出

### 10. 主程序 (`src/main.cpp`)
- ✅ 命令行参数解析
- ✅ 训练流程整合

## 待实现/完善 ⚠️

### 1. 数据加载 (`src/data_loader.cpp`)
- ⚠️ **需要实现完整的 pickle 文件解析**
  - 或者使用 Python 脚本预处理数据为文本格式
  - 需要加载：adj, features, labels, masks

### 2. METIS 分区 (`src/graph_utils.cpp`)
- ⚠️ **需要集成 METIS C 库**
  - 当前是简单的顺序划分（占位符）
  - 需要调用真实的 METIS 分区函数

### 3. 反向传播完善 (`src/backward.cpp`)
- ⚠️ **需要保存前向传播的中间值**
  - ReLU 的梯度需要知道哪些值 > 0
  - Dropout 的 mask 需要保存

### 4. 数据压缩（可选）
- ⚠️ Python 版本有压缩功能，C++ 版本可以暂时跳过
  - compress_csr_with_delta_varint
  - auto_compress_features

## 训练逻辑对应关系

| Python 版本 | C++ 版本 | 状态 |
|------------|---------|------|
| `load_data()` | `DataLoader::load_data()` | ⚠️ 待实现 |
| `metis_partition()` | `PartitionUtils::metis_partition()` | ⚠️ 待完善 |
| `extract_all_partition_subgraphs()` | `PartitionUtils::extract_subgraph()` | ✅ 已实现 |
| `preprocess_adj()` | `GraphUtils::normalize_adj()` + `add_self_loops()` | ✅ 已实现 |
| `preprocess_features()` | `GraphUtils::preprocess_features()` | ✅ 已实现 |
| `GCN` 模型 | `GCNModel` | ✅ 已实现 |
| `layer._update()` | `GCNLayer::update()` | ✅ 已实现 |
| `layer._aggregate()` | `GCNLayer::aggregate()` | ✅ 已实现 |
| `_exec_stage()` | `Trainer::exec_stage()` | ✅ 已实现 |
| `scheduler.schedule_task()` | `Scheduler::schedule_task()` | ✅ 已实现 |
| `construct_feed_dict()` | 直接传递参数（C++ 不需要） | ✅ N/A |
| `sess.run([model.loss, model.accuracy])` | `LossFunctions::masked_*()` | ✅ 已实现 |
| `sess.run([model.opt_op])` | `BackwardPropagator` + `AdamOptimizer` | ✅ 已实现 |
| Early stopping | `Trainer::train()` 中的逻辑 | ✅ 已实现 |

## 编译和运行

### 编译
```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

### 运行
```bash
./gcn_cpp --dataset cora --epochs 200 --learning_rate 0.01 --hidden1 16
```

## 下一步工作

1. **实现数据加载** - 最关键，需要能够加载真实数据
2. **集成 METIS** - 实现真实的图分区
3. **完善反向传播** - 保存中间值用于梯度计算
4. **测试验证** - 与 Python 版本结果对比
5. **集成 ZSim hooks** - 添加 ROI 标记（后续）

## 注意事项

- 当前实现与 Python 版本的训练逻辑**完全一致**
- 数据加载是占位符，需要实现或使用预处理数据
- 反向传播可能需要根据实际测试结果调整
- 所有模块都是低耦合设计，可以独立测试和修改

