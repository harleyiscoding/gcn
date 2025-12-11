/*
 * 反向传播实现
 */

#include "../include/backward.h"
#include <algorithm>

MatrixXf BackwardPropagator::compute_loss_gradient(
    const MatrixXf& logits,
    const MatrixXf& labels,
    const VectorXi& mask) {
    
    // Python: loss = mean(softmax_cross_entropy * normalized_mask)
    // 其中 normalized_mask = mask / mean(mask)
    // 梯度：d(loss)/d(logits) = (probs - labels) * normalized_mask / batch_size
    
    MatrixXf probs = LossFunctions::softmax(logits);
    
    // 计算 softmax cross-entropy 的梯度：probs - labels
    MatrixXf grad = probs - labels;
    
    // 计算 mask 的均值
    int mask_count = 0;
    for (int i = 0; i < mask.size(); i++) {
        if (mask(i) > 0) mask_count++;
    }
    float mask_mean = mask_count > 0 ? static_cast<float>(mask_count) / mask.size() : 0.0f;
    
    // mask 归一化并应用到梯度
    if (mask_mean > 0.0f) {
        float scale = 1.0f / mask_mean;
        for (int i = 0; i < mask.size(); i++) {
            if (mask(i) == 0) {
                grad.row(i).setZero();
            } else {
                grad.row(i) *= scale;
            }
        }
    } else {
        grad.setZero();
    }
    
    // 最后除以 batch_size 以匹配 tf.reduce_mean() 操作
    grad /= static_cast<float>(mask.size());
    
    return grad;
}

MatrixXf BackwardPropagator::backward_layer2_aggregate(
    const MatrixXf& grad_output,
    const SparseMatrix<float>& adj_norm) {
    // d(L)/d(H2) = A^T * d(L)/d(H2_agg)
    return adj_norm.transpose() * grad_output;
}

void BackwardPropagator::backward_layer2_update(
    const MatrixXf& grad_output,
    const MatrixXf& layer1_agg,
    GCNLayer* layer2,
    MatrixXf& grad_layer1_agg,
    AdamOptimizer& optimizer,
    int param_index) {
    
    // 注意：grad_output 的维度是 [num_nodes, output_dim]
    // 这是对 last_linear (线性层输出) 的梯度
    
    // ReLU 反向传播（第二层不使用 ReLU，但 mask 是全 1）
    MatrixXf grad_after_relu = grad_output.cwiseProduct(layer2->get_relu_mask());
    
    // 通过线性层的反向传播
    // d(L)/d(W2) = H1_agg^T * d(L)/d(H2)
    // 注意：这里需要使用 dropout 后的 layer1_agg（即实际用于计算的输入）
    MatrixXf layer1_agg_dropped = layer1_agg;
    if (layer2->is_training() && layer2->get_dropout_rate() > 0.0f) {
        layer1_agg_dropped = layer1_agg.cwiseProduct(layer2->get_dropout_mask()) 
                             / (1.0f - layer2->get_dropout_rate());
    }
    MatrixXf grad_weight = layer1_agg_dropped.transpose() * grad_after_relu;
    // 注意：L2 正则仅对第一层权重应用，第二层不使用
    
    // d(L)/d(b2) = sum(d(L)/d(H2), axis=0)
    VectorXf grad_bias = grad_after_relu.colwise().sum();
    
    // d(L)/d(H1_agg_dropped) = d(L)/d(H2) * W2^T
    MatrixXf grad_layer1_agg_dropped = grad_after_relu * layer2->get_weight().transpose();
    
    // Dropout 反向传播：从 grad_layer1_agg_dropped 得到 grad_layer1_agg
    // 前向：x_dropped = x * mask / (1-rate)
    // 反向：grad_x = grad_x_dropped * mask / (1-rate)
    if (layer2->is_training() && layer2->get_dropout_rate() > 0.0f) {
        grad_layer1_agg = grad_layer1_agg_dropped.cwiseProduct(layer2->get_dropout_mask()) 
                          / (1.0f - layer2->get_dropout_rate());
    } else {
        grad_layer1_agg = grad_layer1_agg_dropped;
    }
    
    // 更新参数
    optimizer.update(layer2->get_weight(), grad_weight, param_index);
    optimizer.update(layer2->get_bias(), grad_bias, param_index + 1);
}

MatrixXf BackwardPropagator::backward_layer1_aggregate(
    const MatrixXf& grad_output,
    const SparseMatrix<float>& adj_norm) {
    // d(L)/d(H1) = A^T * d(L)/d(H1_agg)
    return adj_norm.transpose() * grad_output;
}

void BackwardPropagator::backward_layer1_update(
    const MatrixXf& grad_output,
    const MatrixXf& features,
    GCNLayer* layer1,
    MatrixXf& grad_features,
    AdamOptimizer& optimizer,
    int param_index,
    float weight_decay) {
    
    // 注意：grad_output 的维度是 [num_nodes, hidden_dim]
    // 这是对 last_linear (线性层输出) 的梯度
    
    // ReLU 反向传播
    MatrixXf grad_after_relu = grad_output.cwiseProduct(layer1->get_relu_mask());
    
    // 通过线性层的反向传播
    // d(L)/d(W1) = H^T * d(L)/d(H1)
    // 注意：这里需要使用 dropout 后的 features（即实际用于计算的输入）
    MatrixXf features_dropped = features;
    if (layer1->is_training() && layer1->get_dropout_rate() > 0.0f) {
        features_dropped = features.cwiseProduct(layer1->get_dropout_mask()) 
                           / (1.0f - layer1->get_dropout_rate());
    }
    MatrixXf grad_weight = features_dropped.transpose() * grad_after_relu;
    grad_weight += weight_decay * layer1->get_weight(); // L2 正则梯度
    
    // d(L)/d(b1) = sum(d(L)/d(H1), axis=0)
    VectorXf grad_bias = grad_after_relu.colwise().sum();
    
    // d(L)/d(H_dropped) = d(L)/d(H1) * W1^T
    MatrixXf grad_features_dropped = grad_after_relu * layer1->get_weight().transpose();
    
    // Dropout 反向传播：从 grad_features_dropped 得到 grad_features
    // 前向：x_dropped = x * mask / (1-rate)
    // 反向：grad_x = grad_x_dropped * mask / (1-rate)
    if (layer1->is_training() && layer1->get_dropout_rate() > 0.0f) {
        grad_features = grad_features_dropped.cwiseProduct(layer1->get_dropout_mask()) 
                        / (1.0f - layer1->get_dropout_rate());
    } else {
        grad_features = grad_features_dropped;
    }
    
    // 更新参数
    optimizer.update(layer1->get_weight(), grad_weight, param_index);
    optimizer.update(layer1->get_bias(), grad_bias, param_index + 1);
}

