/*
 * 任务调度器实现
 * 优化版本：支持自适应阈值计算，适应 AMIR/CD 值随时间变化
 */

#include "../include/scheduler.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

// 计算分位数阈值（更稳定，对异常值不敏感）
float Scheduler::compute_percentile_threshold(const std::vector<float>& values, float percentile) {
    if (values.empty()) {
        return 5.0f;  // 默认阈值
    }
    
    if (values.size() == 1) {
        return values[0];
    }
    
    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    // 计算分位数位置
    float index = percentile * (sorted_values.size() - 1);
    int lower = static_cast<int>(std::floor(index));
    int upper = static_cast<int>(std::ceil(index));
    
    if (lower == upper) {
        return sorted_values[lower];
    }
    
    // 线性插值
    float weight = index - lower;
    return sorted_values[lower] * (1.0f - weight) + sorted_values[upper] * weight;
}

// 计算加权阈值（最近的值权重更大）
float Scheduler::compute_weighted_threshold(const std::vector<float>& values) {
    if (values.empty()) {
        return 5.0f;
    }
    
    if (values.size() == 1) {
        return values[0];
    }
    
    // 使用指数衰减权重：w_i = exp(-alpha * (n - i))
    // alpha 控制衰减速度，值越大，最近的值权重越大
    const float alpha = 0.1f;
    float total_weight = 0.0f;
    float weighted_sum = 0.0f;
    
    int n = values.size();
    for (int i = 0; i < n; i++) {
        float weight = std::exp(-alpha * (n - 1 - i));
        weighted_sum += values[i] * weight;
        total_weight += weight;
    }
    
    return weighted_sum / total_weight;
}

// 计算趋势（线性回归斜率）
float Scheduler::compute_trend(const std::vector<float>& values) {
    if (values.size() < 2) {
        return 0.0f;
    }
    
    int n = values.size();
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float x = static_cast<float>(i);
        float y = values[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    float denominator = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-6f) {
        return 0.0f;
    }
    
    // 斜率 = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    return (n * sum_xy - sum_x * sum_y) / denominator;
}

// 自适应阈值计算（结合多种方法）
float Scheduler::compute_adaptive_threshold(const std::vector<float>& values) {
    if (values.empty()) {
        return 5.0f;  // 默认阈值
    }
    
    if (values.size() == 1) {
        return values[0];
    }
    
    // 样本数较少时，使用加权方法（至少需要2个样本）
    if (values.size() == 2) {
        // 对于2个样本，使用加权平均（更重视最新的值）
        return 0.7f * values[1] + 0.3f * values[0];
    }
    
    // 方法1：75分位数（稳定，适合大多数情况）
    float percentile_thresh = compute_percentile_threshold(values, 0.75f);
    
    // 方法2：加权平均（考虑时间衰减）
    float weighted_thresh = compute_weighted_threshold(values);
    
    // 方法3：计算趋势
    float trend = compute_trend(values);
    
    // 如果趋势明显（AMIR 在增长），调整阈值
    // 趋势为正：值在增长，阈值应该相应提高
    float trend_adjustment = 0.0f;
    if (std::abs(trend) > 0.01f) {
        // 根据趋势调整：如果值在增长，阈值也增长
        // 调整幅度 = 趋势 * 窗口大小的一半
        trend_adjustment = trend * values.size() * 0.3f;
    }
    
    // 综合策略：
    // 1. 基础使用加权阈值（考虑最近的值）
    // 2. 结合分位数（避免被异常值影响）
    // 3. 根据趋势调整
    float base_threshold = 0.6f * weighted_thresh + 0.4f * percentile_thresh;
    float adaptive_threshold = base_threshold + trend_adjustment;
    
    // 确保阈值不会过于极端
    float min_val = *std::min_element(values.begin(), values.end());
    float max_val = *std::max_element(values.begin(), values.end());
    adaptive_threshold = std::max(min_val * 0.5f, std::min(max_val * 1.5f, adaptive_threshold));
    
    return adaptive_threshold;
}

// 保留原有的 K-means 方法作为备选
float Scheduler::compute_kmeans_threshold(const std::vector<float>& values) {
    if (values.size() < 2) {
        return 5.0f;  // 默认阈值
    }
    
    // 简化的 K-means（2 聚类）
    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    int mid = sorted_values.size() / 2;
    float center1 = sorted_values[mid / 2];
    float center2 = sorted_values[mid + (sorted_values.size() - mid) / 2];
    
    // 迭代几次
    for (int iter = 0; iter < 10; iter++) {
        float sum1 = 0.0f, sum2 = 0.0f;
        int count1 = 0, count2 = 0;
        
        for (float val : sorted_values) {
            float dist1 = std::abs(val - center1);
            float dist2 = std::abs(val - center2);
            
            if (dist1 < dist2) {
                sum1 += val;
                count1++;
            } else {
                sum2 += val;
                count2++;
            }
        }
        
        if (count1 > 0) center1 = sum1 / count1;
        if (count2 > 0) center2 = sum2 / count2;
    }
    
    return (center1 + center2) / 2.0f;
}

std::string Scheduler::schedule_task(const std::string& phase, float value) {
    if (phase == "AGG") {
        history_amir.push_back(value);
        update_thresholds();
        return (value > amir_threshold) ? "PIM" : "PNM";
    } else if (phase == "UPDATE") {
        history_cd.push_back(value);
        update_thresholds();
        return (value > cd_threshold) ? "GPU" : "PNM";
    } else {
        return "PNM";  // 默认
    }
}

void Scheduler::update_thresholds() {
    // 自适应窗口大小：根据数据量调整
    const int min_window = 5;
    const int max_window = 30;
    
    // 更新 AMIR 阈值
    if (history_amir.size() >= min_window) {
        int window_size = std::min(static_cast<int>(history_amir.size()), max_window);
        window_size = std::max(window_size, min_window);
        
        std::vector<float> recent_amir(
            history_amir.end() - window_size, 
            history_amir.end()
        );
        
        // 使用自适应阈值计算
        float new_threshold = compute_adaptive_threshold(recent_amir);
        
        // 平滑更新：避免阈值变化过于剧烈
        // 使用指数移动平均：threshold = alpha * new + (1-alpha) * old
        const float alpha = 0.3f;  // 平滑系数
        amir_threshold = alpha * new_threshold + (1.0f - alpha) * amir_threshold;
    }
    
    // 更新 CD 阈值
    if (history_cd.size() >= min_window) {
        int window_size = std::min(static_cast<int>(history_cd.size()), max_window);
        window_size = std::max(window_size, min_window);
        
        std::vector<float> recent_cd(
            history_cd.end() - window_size, 
            history_cd.end()
        );
        
        // 使用自适应阈值计算
        float new_threshold = compute_adaptive_threshold(recent_cd);
        
        // 平滑更新
        const float alpha = 0.3f;
        cd_threshold = alpha * new_threshold + (1.0f - alpha) * cd_threshold;
    }
}

void Scheduler::print_statistics() const {
    if (history_amir.empty() && history_cd.empty()) {
        std::cout << "[调度器] 暂无历史数据" << std::endl;
        return;
    }
    
    // 辅助函数：计算趋势
    auto calc_trend = [](const std::vector<float>& vals) -> float {
        if (vals.size() < 2) return 0.0f;
        int n = vals.size();
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
        for (int i = 0; i < n; i++) {
            float x = static_cast<float>(i);
            float y = vals[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        float denominator = n * sum_x2 - sum_x * sum_x;
        if (std::abs(denominator) < 1e-6f) return 0.0f;
        return (n * sum_xy - sum_x * sum_y) / denominator;
    };
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[调度器统计] =================" << std::endl;
    
    if (!history_amir.empty()) {
        float amir_min = *std::min_element(history_amir.begin(), history_amir.end());
        float amir_max = *std::max_element(history_amir.begin(), history_amir.end());
        float amir_mean = std::accumulate(history_amir.begin(), history_amir.end(), 0.0f) / history_amir.size();
        
        // 计算最近值的趋势
        int trend_window = std::min(20, static_cast<int>(history_amir.size()));
        std::vector<float> recent_amir(
            history_amir.end() - trend_window,
            history_amir.end()
        );
        float amir_trend = calc_trend(recent_amir);
        
        std::cout << "AMIR: 阈值=" << amir_threshold 
                  << ", 最小值=" << amir_min
                  << ", 最大值=" << amir_max
                  << ", 平均值=" << amir_mean
                  << ", 趋势=" << amir_trend
                  << ", 样本数=" << history_amir.size() << std::endl;
    }
    
    if (!history_cd.empty()) {
        float cd_min = *std::min_element(history_cd.begin(), history_cd.end());
        float cd_max = *std::max_element(history_cd.begin(), history_cd.end());
        float cd_mean = std::accumulate(history_cd.begin(), history_cd.end(), 0.0f) / history_cd.size();
        
        // 计算最近值的趋势
        int trend_window = std::min(20, static_cast<int>(history_cd.size()));
        std::vector<float> recent_cd(
            history_cd.end() - trend_window,
            history_cd.end()
        );
        float cd_trend = calc_trend(recent_cd);
        
        std::cout << "CD:   阈值=" << cd_threshold
                  << ", 最小值=" << cd_min
                  << ", 最大值=" << cd_max
                  << ", 平均值=" << cd_mean
                  << ", 趋势=" << cd_trend
                  << ", 样本数=" << history_cd.size() << std::endl;
    }
    
    std::cout << "===============================" << std::endl;
}

