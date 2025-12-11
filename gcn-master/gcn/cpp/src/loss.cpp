/*
 * 损失函数实现
 */

#include "../include/loss.h"
#include <algorithm>
#include <numeric>

MatrixXf LossFunctions::softmax(const MatrixXf& logits) {
    // 对每行计算 softmax
    MatrixXf exp_logits = logits;
    
    // 减去最大值以提高数值稳定性
    for (int i = 0; i < exp_logits.rows(); i++) {
        float max_val = exp_logits.row(i).maxCoeff();
        exp_logits.row(i).array() -= max_val;
        exp_logits.row(i) = exp_logits.row(i).array().exp();
        float sum = exp_logits.row(i).sum();
        if (sum > 0) {
            exp_logits.row(i) /= sum;
        }
    }
    
    return exp_logits;
}

float LossFunctions::cross_entropy(const VectorXf& probs, const VectorXf& labels) {
    float loss = 0.0f;
    for (int i = 0; i < probs.size(); i++) {
        if (labels(i) > 0.5f) {
            loss -= std::log(std::max(probs(i), 1e-8f));
        }
    }
    return loss;
}

float LossFunctions::masked_softmax_cross_entropy(
    const MatrixXf& logits,
    const MatrixXf& labels,
    const VectorXi& mask) {
    
    // 与 Python 版本完全一致：
    // 1. 计算 softmax cross-entropy loss
    // 2. mask 归一化：mask /= mean(mask)
    // 3. loss *= mask
    // 4. return mean(loss)
    
    MatrixXf probs = softmax(logits);
    VectorXf loss_per_sample(mask.size());
    
    // 计算每个样本的 cross-entropy loss
    for (int i = 0; i < mask.size(); i++) {
        loss_per_sample(i) = cross_entropy(probs.row(i), labels.row(i));
    }
    
    // 计算 mask 的均值
    int mask_count = 0;
    for (int i = 0; i < mask.size(); i++) {
        if (mask(i) > 0) mask_count++;
    }
    float mask_mean = mask_count > 0 ? static_cast<float>(mask_count) / mask.size() : 0.0f;
    
    // mask 归一化并应用到 loss
    VectorXf normalized_mask = VectorXf::Zero(mask.size());
    if (mask_mean > 0.0f) {
        float scale = 1.0f / mask_mean;
        for (int i = 0; i < mask.size(); i++) {
            if (mask(i) > 0) {
                normalized_mask(i) = scale;
            }
        }
    }
    
    // loss *= normalized_mask
    loss_per_sample = loss_per_sample.cwiseProduct(normalized_mask);
    
    // return mean(loss)
    return loss_per_sample.sum() / mask.size();
}

float LossFunctions::masked_accuracy(
    const MatrixXf& logits,
    const MatrixXf& labels,
    const VectorXi& mask) {
    
    // 与 Python 版本完全一致：
    // 1. 计算 accuracy_all (correct_prediction cast to float)
    // 2. mask 归一化：mask /= mean(mask)
    // 3. accuracy_all *= mask
    // 4. return mean(accuracy_all)
    
    MatrixXf probs = softmax(logits);
    VectorXf accuracy_all = VectorXf::Zero(mask.size());
    
    // 计算每个样本的准确率（1.0 表示正确，0.0 表示错误）
    for (int i = 0; i < mask.size(); i++) {
        // 找到预测的类别
        int pred_class = 0;
        float max_prob = probs(i, 0);
        for (int j = 1; j < probs.cols(); j++) {
            if (probs(i, j) > max_prob) {
                max_prob = probs(i, j);
                pred_class = j;
            }
        }
        
        // 找到真实类别
        int true_class = 0;
        for (int j = 0; j < labels.cols(); j++) {
            if (labels(i, j) > 0.5f) {
                true_class = j;
                break;
            }
        }
        
        accuracy_all(i) = (pred_class == true_class) ? 1.0f : 0.0f;
    }
    
    // 计算 mask 的均值
    int mask_count = 0;
    for (int i = 0; i < mask.size(); i++) {
        if (mask(i) > 0) mask_count++;
    }
    float mask_mean = mask_count > 0 ? static_cast<float>(mask_count) / mask.size() : 0.0f;
    
    // mask 归一化并应用到 accuracy
    VectorXf normalized_mask = VectorXf::Zero(mask.size());
    if (mask_mean > 0.0f) {
        float scale = 1.0f / mask_mean;
        for (int i = 0; i < mask.size(); i++) {
            if (mask(i) > 0) {
                normalized_mask(i) = scale;
            }
        }
    }
    
    // accuracy_all *= normalized_mask
    accuracy_all = accuracy_all.cwiseProduct(normalized_mask);
    
    // return mean(accuracy_all)
    return accuracy_all.sum() / mask.size();
}

float LossFunctions::l2_loss(const MatrixXf& weights, float weight_decay) {
    // 与 TF 的 tf.nn.l2_loss 一致：sum(t^2) / 2
    return 0.5f * weight_decay * weights.cwiseProduct(weights).sum();
}

