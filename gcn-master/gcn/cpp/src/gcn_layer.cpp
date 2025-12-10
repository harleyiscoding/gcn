/*
 * GCN 层实现
 */

#include "../include/gcn_layer.h"
#include <algorithm>
#include <iostream>

// ==================== GCNLayer 实现 ====================

GCNLayer::GCNLayer(int input_dim, int output_dim, bool use_bias, 
                   float dropout, unsigned int seed)
    : use_bias(use_bias), dropout_rate(dropout), training(true),
      rng(seed), dropout_dist(0.0f, 1.0f) {
    initialize_weights(input_dim, output_dim);
    if (use_bias) {
        bias = VectorXf::Zero(output_dim);
    }
}

void GCNLayer::initialize_weights(int input_dim, int output_dim) {
    // Xavier 初始化
    float scale = std::sqrt(2.0f / (input_dim + output_dim));
    weight = MatrixXf::Random(input_dim, output_dim) * scale;
}

MatrixXf GCNLayer::update(const MatrixXf& features) {
    // H' = H * W + b
    MatrixXf output = features * weight;
    
    if (use_bias) {
        output.rowwise() += bias.transpose();
    }
    
    // ReLU 激活
    output = relu(output);
    
    // Dropout（仅在训练时）
    if (training && dropout_rate > 0.0f) {
        output = apply_dropout(output, dropout_rate);
    }
    
    return output;
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
    MatrixXf mask = MatrixXf::Random(x.rows(), x.cols());
    mask = (mask.array() > rate).cast<float>();
    return x.cwiseProduct(mask) / (1.0f - rate);
}

// ==================== GCNModel 实现 ====================

GCNModel::GCNModel(int input_dim, int hidden_dim, int output_dim, float dropout)
    : input_dim(input_dim), hidden_dim(hidden_dim), output_dim(output_dim) {
    layer1 = std::make_unique<GCNLayer>(input_dim, hidden_dim, true, dropout);
    layer2 = std::make_unique<GCNLayer>(hidden_dim, output_dim, true, dropout);
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

