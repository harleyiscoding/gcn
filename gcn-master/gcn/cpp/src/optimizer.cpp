/*
 * 优化器实现
 */

#include "../include/optimizer.h"
#include <algorithm>

// ==================== AdamOptimizer 实现 ====================

void AdamOptimizer::initialize_state(int param_index, int rows, int cols) {
    while (states.size() <= static_cast<size_t>(param_index)) {
        ParamState state;
        state.m = MatrixXf::Zero(rows, cols);
        state.v = MatrixXf::Zero(rows, cols);
        states.push_back(state);
    }
}

void AdamOptimizer::update(MatrixXf& param, const MatrixXf& grad, int param_index) {
    t++;
    
    if (states.size() <= static_cast<size_t>(param_index)) {
        initialize_state(param_index, param.rows(), param.cols());
    }
    
    ParamState& state = states[param_index];
    
    // 更新一阶矩估计
    state.m = beta1 * state.m + (1.0f - beta1) * grad;
    
    // 更新二阶矩估计
    state.v = beta2 * state.v + (1.0f - beta2) * grad.cwiseProduct(grad);
    
    // 偏差修正
    float m_hat_factor = 1.0f / (1.0f - std::pow(beta1, t));
    float v_hat_factor = 1.0f / (1.0f - std::pow(beta2, t));
    
    MatrixXf m_hat = state.m * m_hat_factor;
    MatrixXf v_hat = state.v * v_hat_factor;
    
    // 更新参数
    param -= learning_rate * m_hat.cwiseQuotient(
        v_hat.cwiseSqrt() + MatrixXf::Constant(v_hat.rows(), v_hat.cols(), epsilon)
    );
}

void AdamOptimizer::update(VectorXf& param, const VectorXf& grad, int param_index) {
    t++;
    
    if (states.size() <= static_cast<size_t>(param_index)) {
        initialize_state(param_index, param.size(), 1);
    }
    
    ParamState& state = states[param_index];
    
    // 将 VectorXf 转换为 MatrixXf 进行处理
    MatrixXf param_mat = param;
    MatrixXf grad_mat = grad;
    MatrixXf m_mat = state.m.col(0);
    MatrixXf v_mat = state.v.col(0);
    
    // 更新
    m_mat = beta1 * m_mat + (1.0f - beta1) * grad_mat;
    v_mat = beta2 * v_mat + (1.0f - beta2) * grad_mat.cwiseProduct(grad_mat);
    
    float m_hat_factor = 1.0f / (1.0f - std::pow(beta1, t));
    float v_hat_factor = 1.0f / (1.0f - std::pow(beta2, t));
    
    MatrixXf m_hat = m_mat * m_hat_factor;
    MatrixXf v_hat = v_mat * v_hat_factor;
    
    param_mat -= learning_rate * m_hat.cwiseQuotient(
        v_hat.cwiseSqrt() + MatrixXf::Constant(v_hat.rows(), v_hat.cols(), epsilon)
    );
    
    param = param_mat.col(0);
    
    // 更新状态
    state.m.col(0) = m_mat;
    state.v.col(0) = v_mat;
}

void AdamOptimizer::reset() {
    states.clear();
    t = 0;
}

// ==================== SGDOptimizer 实现 ====================

void SGDOptimizer::initialize_state(int param_index, int rows, int cols) {
    while (states.size() <= static_cast<size_t>(param_index)) {
        ParamState state;
        state.velocity = MatrixXf::Zero(rows, cols);
        states.push_back(state);
    }
}

void SGDOptimizer::update(MatrixXf& param, const MatrixXf& grad, int param_index) {
    if (states.size() <= static_cast<size_t>(param_index)) {
        initialize_state(param_index, param.rows(), param.cols());
    }
    
    ParamState& state = states[param_index];
    
    if (momentum > 0.0f) {
        state.velocity = momentum * state.velocity + learning_rate * grad;
        param -= state.velocity;
    } else {
        param -= learning_rate * grad;
    }
}

void SGDOptimizer::update(VectorXf& param, const VectorXf& grad, int param_index) {
    if (states.size() <= static_cast<size_t>(param_index)) {
        initialize_state(param_index, param.size(), 1);
    }
    
    ParamState& state = states[param_index];
    MatrixXf velocity_vec = state.velocity.col(0);
    MatrixXf param_mat = param;
    MatrixXf grad_mat = grad;
    
    if (momentum > 0.0f) {
        velocity_vec = momentum * velocity_vec + learning_rate * grad_mat;
        param_mat -= velocity_vec;
    } else {
        param_mat -= learning_rate * grad_mat;
    }
    
    param = param_mat.col(0);
    state.velocity.col(0) = velocity_vec;
}

void SGDOptimizer::reset() {
    states.clear();
}

