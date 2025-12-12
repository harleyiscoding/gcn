/*
 * 数据类型定义
 * 定义 GCN 训练中使用的核心数据结构
 */

#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <string>
#include <memory>

using namespace Eigen;

// ==================== 配置结构 ====================

struct GCNConfig {
    std::string dataset = "cora";
    std::string model = "gcn";  // "gcn", "gcn_cheby", "dense"
    float learning_rate = 0.01f;
    int epochs = 200;
    int hidden1 = 16;
    float dropout = 0.5f;
    float weight_decay = 5e-4f;
    int early_stopping = 200;              // 早停容忍度：与 Python 版本一致
                                          // 从第 (early_stopping + 1) 个 epoch 开始检查，如果当前验证损失大于最近 early_stopping 个 epoch 的平均值则停止
                                          // 设置为 >= epochs 的值可禁用早停（如 200 表示训练 200 个 epoch 时不会触发早停）
    int max_degree = 3;
    int num_parts = 1;
    int seed = 123;                         // 与 Python 对齐的全局随机种子
    bool enable_roi_marking = false;  // 暂时不使用
};

// ==================== 图数据结构 ====================

struct GraphData {
    SparseMatrix<float> adj;              // 邻接矩阵（CSR 格式）
    MatrixXf features;                     // 特征矩阵（稠密）
    MatrixXf labels;                        // 标签矩阵（one-hot）
    VectorXi train_mask;                   // 训练集掩码
    VectorXi val_mask;                     // 验证集掩码
    VectorXi test_mask;                    // 测试集掩码
    
    // 三元组表示（用于稀疏特征）
    struct SparseTuple {
        MatrixXi coords;                   // 坐标矩阵 [N, 2]
        VectorXf values;                   // 值向量
        std::pair<int, int> shape;         // (rows, cols)
    };
    SparseTuple features_tuple;            // 特征的三元组表示
    
    int num_nodes;
    int num_features;
    int num_classes;
};

// ==================== 子图数据结构 ====================

struct SubgraphData {
    SparseMatrix<float> adj_sub;           // 子图邻接矩阵
    MatrixXf features_sub;                  // 子图特征矩阵
    MatrixXf y_train_sub;                  // 子图训练标签
    MatrixXf y_val_sub;                    // 子图验证标签
    MatrixXf y_test_sub;                    // 子图测试标签
    VectorXi train_mask_sub;               // 子图训练掩码
    VectorXi val_mask_sub;                 // 子图验证掩码
    VectorXi test_mask_sub;                // 子图测试掩码
    VectorXi part_nodes;                   // 子图节点在全图中的索引
    
    // 压缩数据（可选）
    std::vector<int> adj_compressed_indptr;
    std::vector<uint8_t> adj_compressed_indices;
    std::vector<float> adj_compressed_data;
    std::pair<int, int> adj_shape;
    
    // 预处理后的数据
    GraphData::SparseTuple features_tuple; // 预处理后的特征三元组
    std::vector<SparseMatrix<float>> support; // 支持矩阵列表（归一化后的邻接矩阵）
    
    int num_nodes() const { return adj_sub.rows(); }
};

// ==================== 任务信息结构 ====================

struct TaskInfo {
    int layer;                              // 层索引 (1 or 2)
    std::string phase;                      // "UPDATE" or "AGG"
    float value;                            // AMIR 或 CD 值
};

// ==================== 阶段日志结构 ====================

struct StageLog {
    int epoch;
    int stage;                              // 阶段索引 (1-4)
    int layer;
    std::string phase;
    float value;
    int partition_id;
    std::string device;                     // "PIM", "PNM", "GPU"
};

// ==================== 训练统计 ====================

struct TrainingStats {
    float train_loss = 0.0f;
    float train_acc = 0.0f;
    float val_loss = 0.0f;
    float val_acc = 0.0f;
    double time_elapsed = 0.0;
};

#endif // TYPES_H

