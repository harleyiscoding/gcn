/*
 * 反向传播实现
 */

#include "../include/backward.h"
#include <algorithm>

MatrixXf BackwardPropagator::compute_loss_gradient(
    const MatrixXf& logits,
    const MatrixXf& labels,
    const VectorXi& mask) {
    
    // Softmax 输出
    MatrixXf probs = LossFunctions::softmax(logits);
    
    // 计算梯度：probs - labels（对于掩码内的节点）
    MatrixXf grad = probs - labels;
    
    // 应用掩码
    for (int i = 0; i < mask.size(); i++) {
        if (mask(i) == 0) {
            grad.row(i).setZero();
        }
    }
    
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
    
    // 通过 ReLU 的反向传播
    MatrixXf grad_before_relu = grad_output;
    // TODO: 需要保存前向传播时的激活值来判断 ReLU 的梯度
    
    // 通过线性层的反向传播
    // d(L)/d(W2) = H1_agg^T * d(L)/d(H2)
    MatrixXf grad_weight = layer1_agg.transpose() * grad_before_relu;
    
    // d(L)/d(b2) = sum(d(L)/d(H2), axis=0)
    VectorXf grad_bias = grad_before_relu.colwise().sum();
    
    // d(L)/d(H1_agg) = d(L)/d(H2) * W2^T
    grad_layer1_agg = grad_before_relu * layer2->get_weight().transpose();
    
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
    int param_index) {
    
    // 通过 ReLU 的反向传播
    MatrixXf grad_before_relu = grad_output;
    // TODO: 需要保存前向传播时的激活值
    
    // 通过线性层的反向传播
    // d(L)/d(W1) = H^T * d(L)/d(H1)
    MatrixXf grad_weight = features.transpose() * grad_before_relu;
    
    // d(L)/d(b1) = sum(d(L)/d(H1), axis=0)
    VectorXf grad_bias = grad_before_relu.colwise().sum();
    
    // d(L)/d(H) = d(L)/d(H1) * W1^T
    grad_features = grad_before_relu * layer1->get_weight().transpose();
    
    // 更新参数
    optimizer.update(layer1->get_weight(), grad_weight, param_index);
    optimizer.update(layer1->get_bias(), grad_bias, param_index + 1);
}

