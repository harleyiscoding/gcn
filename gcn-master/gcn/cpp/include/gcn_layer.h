/*
 * GCN 层实现
 * 实现 Graph Convolutional Network 的层，包含 Update 和 Aggregate 操作
 */

#ifndef GCN_LAYER_H
#define GCN_LAYER_H

#include "types.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <cmath>

// ==================== GCN 层基类 ====================

class GCNLayer {
protected:
    MatrixXf weight;                        // 权重矩阵 W [input_dim, output_dim]
    VectorXf bias;                          // 偏置向量
    bool use_bias;
    float dropout_rate;
    bool training;
    
    // 随机数生成器（用于 dropout）
    std::mt19937 rng;
    std::uniform_real_distribution<float> dropout_dist;
    
public:
    GCNLayer(int input_dim, int output_dim, bool use_bias = true, 
             float dropout = 0.0f, unsigned int seed = 42);
    
    virtual ~GCNLayer() = default;
    
    // Update 操作：H' = H * W + b，然后 ReLU
    // 输入：features [num_nodes, input_dim]
    // 输出：updated [num_nodes, output_dim]
    MatrixXf update(const MatrixXf& features);
    
    // Aggregate 操作：H'' = A_norm * H'
    // 输入：adj_norm [num_nodes, num_nodes] (稀疏), features [num_nodes, dim]
    // 输出：aggregated [num_nodes, dim]
    MatrixXf aggregate(const SparseMatrix<float>& adj_norm, const MatrixXf& features);
    
    // 设置训练模式
    void set_training(bool is_training) { training = is_training; }
    
    // 获取/设置权重（用于反向传播）
    MatrixXf& get_weight() { return weight; }
    VectorXf& get_bias() { return bias; }
    const MatrixXf& get_weight() const { return weight; }
    const VectorXf& get_bias() const { return bias; }
    
    // 初始化权重（Xavier 初始化）
    void initialize_weights(int input_dim, int output_dim);
    
private:
    // ReLU 激活函数
    MatrixXf relu(const MatrixXf& x);
    
    // Dropout
    MatrixXf apply_dropout(const MatrixXf& x, float rate);
};

// ==================== GCN 模型（两层） ====================

class GCNModel {
private:
    std::unique_ptr<GCNLayer> layer1;       // 第一层
    std::unique_ptr<GCNLayer> layer2;       // 第二层
    int input_dim;
    int hidden_dim;
    int output_dim;
    
public:
    GCNModel(int input_dim, int hidden_dim, int output_dim, 
             float dropout = 0.5f);
    
    // 前向传播：执行完整的 GCN 前向传播
    // 返回：logits [num_nodes, output_dim]
    MatrixXf forward(const SparseMatrix<float>& adj_norm, 
                     const MatrixXf& features);
    
    // 分阶段前向传播（用于调度）
    // Layer 1 Update
    MatrixXf layer1_update(const MatrixXf& features);
    
    // Layer 1 Aggregate
    MatrixXf layer1_aggregate(const SparseMatrix<float>& adj_norm, 
                               const MatrixXf& layer1_output);
    
    // Layer 2 Update
    MatrixXf layer2_update(const MatrixXf& layer1_agg);
    
    // Layer 2 Aggregate
    MatrixXf layer2_aggregate(const SparseMatrix<float>& adj_norm, 
                               const MatrixXf& layer2_output);
    
    // 设置训练模式
    void set_training(bool is_training);
    
    // 获取层（用于反向传播）
    GCNLayer* get_layer1() { return layer1.get(); }
    GCNLayer* get_layer2() { return layer2.get(); }
};

#endif // GCN_LAYER_H

