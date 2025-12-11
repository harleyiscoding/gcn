/*
 * GCN 层实现
 */

#include "../include/gcn_layer.h"
#include <algorithm>
#include <iostream>

// ==================== GCNLayer 实现 ====================

GCNLayer::GCNLayer(int input_dim, int output_dim, std::mt19937& rng_ref,
                   bool use_bias, float dropout, bool use_relu)
    : use_bias(use_bias), dropout_rate(dropout), training(true), use_relu(use_relu),
      rng(rng_ref), dropout_dist(0.0f, 1.0f) {
    initialize_weights(input_dim, output_dim);
    if (use_bias) {
        bias = VectorXf::Zero(output_dim);
    }
}

void GCNLayer::initialize_weights(int input_dim, int output_dim) {
    // Glorot Uniform 初始化（与 TF 一致）
    float limit = std::sqrt(6.0f / (input_dim + output_dim));
    std::uniform_real_distribution<float> dist(-limit, limit);
    weight.resize(input_dim, output_dim);
    for (int i = 0; i < input_dim; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            weight(i, j) = dist(rng);
        }
    }
}

MatrixXf GCNLayer::update(const MatrixXf& features) {
    // 与 Python GraphConvolution 完全一致：
    // 1. dropout 输入（在 transform 之前）
    // 2. transform: H' = H * W + b
    // 3. ReLU 激活（如果启用）
    // 注意：Python 中 dropout 是在输入上做的，不是输出
    
    MatrixXf x = features;
    
    // Dropout 输入（仅在训练时）
    if (training && dropout_rate > 0.0f) {
        x = apply_dropout(x, dropout_rate);
    } else {
        // 推理或未启用 dropout 时，使用单位 mask，便于反向传播
        dropout_mask = MatrixXf::Ones(x.rows(), x.cols());
    }
    
    // Transform: H' = H * W + b
    last_linear = x * weight;
    
    if (use_bias) {
        last_linear.rowwise() += bias.transpose();
    }
    
    // ReLU 激活（如果启用，保存 mask 用于反向传播）
    if (use_relu) {
        relu_mask = (last_linear.array() > 0.0f).cast<float>();
        MatrixXf output = relu(last_linear);
        return output;
    } else {
        // 第二层不使用 ReLU（identity activation）
        relu_mask = MatrixXf::Ones(last_linear.rows(), last_linear.cols());
        return last_linear;
    }
}

MatrixXf GCNLayer::aggregate(const SparseMatrix<float>& adj_norm, 
                             const MatrixXf& features) {
    // H'' = A_norm * H'
    return adj_norm * features;
}

MatrixXf GCNLayer::relu(const MatrixXf& x) {
    return x.cwiseMax(0.0f);
}

MatrixXf GCNLayer::apply_dropout(const MatrixXf& x, float rate) {
    if (rate <= 0.0f) {
        return x;
    }
    
    dropout_mask.resize(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            dropout_mask(i, j) = (dropout_dist(rng) > rate) ? 1.0f : 0.0f;
        }
    }
    // 反向传播时也复用同一个 mask
    return x.cwiseProduct(dropout_mask) / (1.0f - rate);
}

// ==================== GCNModel 实现 ====================

GCNModel::GCNModel(int input_dim, int hidden_dim, int output_dim, float dropout, unsigned int seed)
    : input_dim(input_dim), hidden_dim(hidden_dim), output_dim(output_dim),
      global_rng(seed) {  // 初始化全局随机数生成器
    // 所有层共享同一个全局 RNG（与 Python 版本一致）
    // Layer 1: 使用 ReLU 激活
    layer1 = std::make_unique<GCNLayer>(input_dim, hidden_dim, global_rng, true, dropout, true);
    // Layer 2: 不使用 ReLU（identity activation，与 Python 的 act=lambda x: x 一致）
    layer2 = std::make_unique<GCNLayer>(hidden_dim, output_dim, global_rng, true, dropout, false);
}

MatrixXf GCNModel::forward(const SparseMatrix<float>& adj_norm, 
                          const MatrixXf& features) {
    // 完整前向传播
    MatrixXf h1 = layer1_update(features);
    MatrixXf h1_agg = layer1_aggregate(adj_norm, h1);
    MatrixXf h2 = layer2_update(h1_agg);
    MatrixXf logits = layer2_aggregate(adj_norm, h2);
    return logits;
}

MatrixXf GCNModel::layer1_update(const MatrixXf& features) {
    return layer1->update(features);
}

MatrixXf GCNModel::layer1_aggregate(const SparseMatrix<float>& adj_norm, 
                                   const MatrixXf& layer1_output) {
    return layer1->aggregate(adj_norm, layer1_output);
}

MatrixXf GCNModel::layer2_update(const MatrixXf& layer1_agg) {
    return layer2->update(layer1_agg);
}

MatrixXf GCNModel::layer2_aggregate(const SparseMatrix<float>& adj_norm, 
                                   const MatrixXf& layer2_output) {
    return layer2->aggregate(adj_norm, layer2_output);
}

void GCNModel::set_training(bool is_training) {
    layer1->set_training(is_training);
    layer2->set_training(is_training);
}

