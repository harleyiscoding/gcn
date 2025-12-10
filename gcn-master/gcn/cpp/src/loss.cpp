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
    
    MatrixXf probs = softmax(logits);
    float total_loss = 0.0f;
    int count = 0;
    
    for (int i = 0; i < mask.size(); i++) {
        if (mask(i) > 0) {
            total_loss += cross_entropy(probs.row(i), labels.row(i));
            count++;
        }
    }
    
    return count > 0 ? total_loss / count : 0.0f;
}

float LossFunctions::masked_accuracy(
    const MatrixXf& logits,
    const MatrixXf& labels,
    const VectorXi& mask) {
    
    MatrixXf probs = softmax(logits);
    int correct = 0;
    int total = 0;
    
    for (int i = 0; i < mask.size(); i++) {
        if (mask(i) > 0) {
            total++;
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
            
            if (pred_class == true_class) {
                correct++;
            }
        }
    }
    
    return total > 0 ? static_cast<float>(correct) / total : 0.0f;
}

float LossFunctions::l2_loss(const MatrixXf& weights, float weight_decay) {
    return weight_decay * weights.cwiseProduct(weights).sum();
}

