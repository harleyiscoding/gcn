# GCN C++ 实现

这是 GCN 训练代码的 C++ 版本实现，使用 Eigen 库替代 TensorFlow，直接集成 ZSim hooks 用于 trace 生成。

## 目录结构

```
cpp/
├── README.md                    # 本文件
├── GCN_CPP_FEASIBILITY.md      # 可行性分析文档
├── gcn_cpp_stub.cpp            # C++ 实现框架代码
├── CMakeLists.txt              # CMake 构建配置（待创建）
└── src/                        # 源代码目录（待创建）
    ├── data_loader.cpp
    ├── graph_utils.cpp
    ├── gcn_layer.cpp
    └── scheduler.cpp
```

## 编译说明

### 依赖项

- **Eigen3**: 矩阵运算库
- **METIS**: 图分区库（可选）
- **OpenMP**: 并行计算支持
- **ZSim hooks**: 用于 ROI 标记

### 编译命令

```bash
# 基本编译
g++ -std=c++17 -O3 -fopenmp gcn_cpp_stub.cpp -o gcn_cpp \
    -I/path/to/eigen \
    -I../../../ramulator-pim-master/zsim-ramulator/misc/hooks \
    -lmetis

# 或使用 CMake（推荐）
mkdir build && cd build
cmake ..
make
```

## 使用方法

```bash
# 运行训练
./gcn_cpp cora 200 true

# 参数说明：
# 1. dataset: 数据集名称（cora, citeseer, pubmed）
# 2. epochs: 训练轮数
# 3. enable_roi_marking: 是否启用 ROI 标记（true/false）
```

## 与 Python 版本的对应关系

| Python 模块 | C++ 文件 | 说明 |
|------------|---------|------|
| `task_scheduler_distributed.py` | `gcn_cpp_stub.cpp` | 主训练逻辑 |
| `models.py` | `gcn_layer.cpp` | GCN 层实现 |
| `utils.py` | `graph_utils.cpp` | 图处理工具 |
| `partition_utils.py` | `graph_utils.cpp` | 图分区相关 |
| - | `data_loader.cpp` | 数据加载 |
| - | `scheduler.cpp` | 调度器实现 |

## 开发状态

- [x] 框架代码结构
- [ ] 数据加载实现
- [ ] 图预处理实现
- [ ] GCN 层完整实现
- [ ] 反向传播实现
- [ ] 优化器实现
- [ ] 调度器完整实现
- [ ] ZSim 集成测试

## 注意事项

1. 当前代码为框架/占位符，需要逐步实现各个模块
2. ZSim hooks 路径需要根据实际项目结构调整
3. Eigen 库需要单独下载或通过包管理器安装

## 参考文档

- [可行性分析](GCN_CPP_FEASIBILITY.md)
- [Eigen 文档](https://eigen.tuxfamily.org/)
- [ZSim 文档](../../../ramulator-pim-master/zsim-ramulator/README.md)

