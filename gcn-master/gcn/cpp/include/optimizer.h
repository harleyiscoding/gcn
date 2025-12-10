/*
 * 优化器实现
 * 实现 SGD 和 Adam 优化器
 */

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>

using namespace Eigen;

// ==================== Adam 优化器 ====================

class AdamOptimizer {
private:
    float learning_rate;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int t = 0;  // 时间步
    
    // 为每个参数维护 m 和 v
    struct ParamState {
        MatrixXf m;  // 一阶矩估计
        MatrixXf v;  // 二阶矩估计
    };
    
    std::vector<ParamState> states;
    
public:
    AdamOptimizer(float lr) : learning_rate(lr) {}
    
    // 初始化参数状态
    void initialize_state(int param_index, int rows, int cols);
    
    // 更新参数
    void update(MatrixXf& param, const MatrixXf& grad, int param_index);
    void update(VectorXf& param, const VectorXf& grad, int param_index);
    
    // 重置（用于新模型）
    void reset();
};

// ==================== SGD 优化器 ====================

class SGDOptimizer {
private:
    float learning_rate;
    float momentum = 0.0f;
    
    struct ParamState {
        MatrixXf velocity;
    };
    
    std::vector<ParamState> states;
    
public:
    SGDOptimizer(float lr, float momentum = 0.0f) 
        : learning_rate(lr), momentum(momentum) {}
    
    void initialize_state(int param_index, int rows, int cols);
    void update(MatrixXf& param, const MatrixXf& grad, int param_index);
    void update(VectorXf& param, const VectorXf& grad, int param_index);
    void reset();
};

#endif // OPTIMIZER_H

