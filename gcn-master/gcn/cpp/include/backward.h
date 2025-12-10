/*
 * 反向传播实现
 * 计算梯度并更新参数
 */

#ifndef BACKWARD_H
#define BACKWARD_H

#include "gcn_layer.h"
#include "loss.h"
#include "optimizer.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

// ==================== 反向传播器 ====================

class BackwardPropagator {
public:
    // 计算损失对输出的梯度
    static MatrixXf compute_loss_gradient(
        const MatrixXf& logits,
        const MatrixXf& labels,
        const VectorXi& mask);
    
    // Layer 2 Aggregate 反向传播
    static MatrixXf backward_layer2_aggregate(
        const MatrixXf& grad_output,
        const SparseMatrix<float>& adj_norm);
    
    // Layer 2 Update 反向传播
    static void backward_layer2_update(
        const MatrixXf& grad_output,
        const MatrixXf& layer1_agg,
        GCNLayer* layer2,
        MatrixXf& grad_layer1_agg,
        AdamOptimizer& optimizer,
        int param_index);
    
    // Layer 1 Aggregate 反向传播
    static MatrixXf backward_layer1_aggregate(
        const MatrixXf& grad_output,
        const SparseMatrix<float>& adj_norm);
    
    // Layer 1 Update 反向传播
    static void backward_layer1_update(
        const MatrixXf& grad_output,
        const MatrixXf& features,
        GCNLayer* layer1,
        MatrixXf& grad_features,
        AdamOptimizer& optimizer,
        int param_index);
};

#endif // BACKWARD_H

