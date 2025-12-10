/*
 * 损失函数和评估指标
 */

#ifndef LOSS_H
#define LOSS_H

#include "types.h"
#include <Eigen/Dense>
#include <cmath>

// ==================== 损失函数 ====================

class LossFunctions {
public:
    // 交叉熵损失（带掩码）
    static float masked_softmax_cross_entropy(
        const MatrixXf& logits,
        const MatrixXf& labels,
        const VectorXi& mask);
    
    // 准确率（带掩码）
    static float masked_accuracy(
        const MatrixXf& logits,
        const MatrixXf& labels,
        const VectorXi& mask);
    
    // L2 正则化损失
    static float l2_loss(const MatrixXf& weights, float weight_decay);
    
    // Softmax（public，供反向传播使用）
    static MatrixXf softmax(const MatrixXf& logits);
    
private:
    // 计算交叉熵
    static float cross_entropy(const VectorXf& probs, const VectorXf& labels);
};

#endif // LOSS_H

